package store_test

/*
 * ============================================================================
 * ARCHITECTURAL INVARIANT: SPLIT-BOUNDARY LOCK ROUTING & RESOLUTION
 * ============================================================================
 *
 * Core Invariant:
 * When an existing Region R1 [StartKey, EndKey) splits into R1' [StartKey, SplitKey)
 * and child Region R2 [SplitKey, EndKey), all active lock operations (commit,
 * rollback, pessimistic unlock, and lock resolution) targeting keys >= SplitKey
 * must be transferred and re-routed cleanly to R2.
 *
 * Failure Mode / Race Condition:
 * 1. A client retains a cached descriptor for pre-split R1 [StartKey, EndKey).
 * 2. Multi-key rollback or secondary commit commands for keys >= SplitKey route
 *    to R1 based on stale RegionCache mappings.
 * 3. TiKV rejects requests covering keys outside R1' with `EpochNotMatch`.
 * 4. If RegionCache fails to atomically drop the orphaned range [SplitKey, EndKey),
 *    or if callers fail to re-locate every key in the batch individually:
 *      - Secondary locks on R2 are stranded uncleaned.
 *      - Concurrent transactions block indefinitely or until lock TTL expiration.
 *
 * Required Behavior:
 * - `RegionCache.OnRegionEpochNotMatch` must invalidate the parent region's range.
 * - Retry routines must re-resolve keys >= SplitKey to R2 via `current_regions`
 *   or PD lookup before retrying lock cleanup.
 * ============================================================================
 */

import (
	"context"
	"testing"

	"github.com/pingcap/tidb/pkg/kv"
	"github.com/pingcap/tidb/pkg/store/mockstore"
	"github.com/stretchr/testify/require"
	tikv "github.com/tikv/client-go/v2/tikv"
	"github.com/tikv/client-go/v2/tikv/mockstore/cluster"
)

type tikvStoreWrapper interface {
	GetTiKVStore() *tikv.KVStore
}

func TestPessimisticLockResolutionAfterRegionSplit(t *testing.T) {
	var mockCluster cluster.Cluster
	store, err := mockstore.NewMockStore(
		mockstore.WithClusterInspector(func(c cluster.Cluster) {
			mockCluster = c
		}),
	)
	require.NoError(t, err)
	defer store.Close()

	ctx := context.Background()

	// Keys spanning across the planned split point
	k1 := kv.Key("k_001_primary")
	k2 := kv.Key("k_002_secondary")
	val1 := []byte("v1")
	val2 := []byte("v2")

	// 1. Acquire pessimistic locks on both keys in Region 1
	txn1, err := store.Begin()
	require.NoError(t, err)
	err = txn1.SetOption(kv.Pessimistic, true)
	require.NoError(t, err)

	require.NoError(t, txn1.Set(k1, val1))
	require.NoError(t, txn1.Set(k2, val2))

	lockCtx := &kv.LockCtx{ForUpdateTS: txn1.StartTS()}
	err = txn1.LockKeys(ctx, lockCtx, k1, k2)
	require.NoError(t, err)

	// 2. Trigger region split at k2 (R1 shrinks to ["", k2), R2 takes [k2, ""))
	origRegion := mockCluster.GetRegionByKey(k2)
	require.NotNil(t, origRegion)

	newRegionID, err := mockCluster.AllocID()
	require.NoError(t, err)
	newPeerID, err := mockCluster.AllocID()
	require.NoError(t, err)

	mockCluster.Split(origRegion.GetId(), newRegionID, k2, []uint64{newPeerID}, newPeerID)

	// 3. Roll back Txn1: client-go must unlock k2 on child region R2 despite stale cache
	require.NoError(t, txn1.Rollback())

	// 4. Assert k2 lock was released by successfully locking it in a new transaction
	txn2, err := store.Begin()
	require.NoError(t, err)
	defer txn2.Rollback()

	err = txn2.SetOption(kv.Pessimistic, true)
	require.NoError(t, err)

	lockCtx2 := &kv.LockCtx{ForUpdateTS: txn2.StartTS()}
	err = txn2.LockKeys(ctx, lockCtx2, k2)
	require.NoError(t, err, "k2 lock was not cleared on child region after split rollback")

	// 5. Verify cache reloaded R2
	if wrapper, ok := store.(tikvStoreWrapper); ok {
		tikvStore := wrapper.GetTiKVStore()
		rc := tikvStore.GetRegionCache()
		loc, err := rc.LocateKey(tikv.NewBackofferWithVars(ctx, 1000, nil), k2)
		require.NoError(t, err)
		require.Equal(t, newRegionID, loc.Region.GetID(), "RegionCache did not reload child region")
	}
}

func TestCommitSecondaryLockAfterRegionSplit(t *testing.T) {
	var mockCluster cluster.Cluster
	store, err := mockstore.NewMockStore(
		mockstore.WithClusterInspector(func(c cluster.Cluster) {
			mockCluster = c
		}),
	)
	require.NoError(t, err)
	defer store.Close()

	ctx := context.Background()

	k1 := kv.Key("k_100_primary")
	k2 := kv.Key("k_200_secondary")
	val1 := []byte("v100")
	val2 := []byte("v200")

	// 1. Acquire pessimistic locks across the boundary
	txn1, err := store.Begin()
	require.NoError(t, err)
	err = txn1.SetOption(kv.Pessimistic, true)
	require.NoError(t, err)

	require.NoError(t, txn1.Set(k1, val1))
	require.NoError(t, txn1.Set(k2, val2))

	lockCtx := &kv.LockCtx{ForUpdateTS: txn1.StartTS()}
	err = txn1.LockKeys(ctx, lockCtx, k1, k2)
	require.NoError(t, err)

	// 2. Split region between k1 and k2
	origRegion := mockCluster.GetRegionByKey(k2)
	require.NotNil(t, origRegion)

	newRegionID, err := mockCluster.AllocID()
	require.NoError(t, err)
	newPeerID, err := mockCluster.AllocID()
	require.NoError(t, err)

	mockCluster.Split(origRegion.GetId(), newRegionID, k2, []uint64{newPeerID}, newPeerID)

	// 3. Commit Txn1: secondary commit must route to child region R2
	require.NoError(t, txn1.Commit(ctx))

	// 4. Confirm committed value is readable
	txn2, err := store.Begin()
	require.NoError(t, err)
	defer txn2.Rollback()

	gotVal, err := txn2.Get(ctx, k2)
	require.NoError(t, err)
	require.Equal(t, val2, gotVal)
}
