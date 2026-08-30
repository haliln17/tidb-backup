package store_test

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

// =============================================================================
// INVARIANT DOCUMENTATION: REGION SPLIT LOCK ROUTING & RESOLUTION
//
// Invariant:
// When an existing Region R1 [StartKey, EndKey) splits into R1' [StartKey, SplitKey)
// (with an incremented epoch version) and a new child Region R2 [SplitKey, EndKey),
// lock lifecycle operations (Commit, Rollback, Pessimistic Unlock, ResolveLocks)
// addressing keys >= SplitKey MUST NOT be dropped, skipped, or misrouted.
//
// Client-go Execution Path:
// 1. A client holding a stale RegionCache entry for [StartKey, EndKey) dispatches
//    a lock resolution or 2PC secondary commit/rollback RPC to R1.
// 2. TiKV rejects the request with an `EpochNotMatch` error.
// 3. RegionCache.OnRegionEpochNotMatch must invalidate the stale descriptor for R1.
// 4. The retry loop (via Backoffer) must re-locate all keys >= SplitKey to R2
//    (discovering R2 either from the error payload's `current_regions` or PD).
// 5. The secondary lock on R2 must be explicitly cleared to prevent leaving
//    orphaned secondary locks on the child region that block concurrent writers
//    until TTL expiry.
// =============================================================================

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

	// Keys designed to start in the same initial region ["", "")
	k1 := kv.Key("k_001_primary")
	k2 := kv.Key("k_002_secondary")
	val1 := []byte("v1")
	val2 := []byte("v2")

	// 1. Start Txn1 and acquire pessimistic locks on both keys in Region 1
	txn1, err := store.Begin()
	require.NoError(t, err)
	err = txn1.SetOption(kv.Pessimistic, true)
	require.NoError(t, err)

	require.NoError(t, txn1.Set(k1, val1))
	require.NoError(t, txn1.Set(k2, val2))

	lockCtx := &kv.LockCtx{ForUpdateTS: txn1.StartTS()}
	err = txn1.LockKeys(ctx, lockCtx, k1, k2)
	require.NoError(t, err)

	// 2. Trigger a Region split at k2 on the cluster.
	// Region 1 shrinks to ["", "k_002_secondary").
	// Region 2 is created covering ["k_002_secondary", "").
	origRegion := mockCluster.GetRegionByKey(k2)
	require.NotNil(t, origRegion)

	newRegionID, err := mockCluster.AllocID()
	require.NoError(t, err)
	newPeerID, err := mockCluster.AllocID()
	require.NoError(t, err)

	mockCluster.Split(origRegion.GetId(), newRegionID, k2, []uint64{newPeerID}, newPeerID)

	/*
	 * INVARIANT ENFORCEMENT POINT:
	 * Txn1 rolls back, requiring client-go to release pessimistic locks across
	 * both keys. k1 is still in R1, but k2 now resides in R2.
	 * client-go's LockResolver / region cache must handle EpochNotMatch on k2,
	 * re-route the unlock request to newRegionID, and avoid stranding k2's lock.
	 */
	require.NoError(t, txn1.Rollback())

	// 3. Start Txn2 and assert that k2 is immediately lockable without blocking on a stale lock
	txn2, err := store.Begin()
	require.NoError(t, err)
	defer txn2.Rollback()

	err = txn2.SetOption(kv.Pessimistic, true)
	require.NoError(t, err)

	lockCtx2 := &kv.LockCtx{ForUpdateTS: txn2.StartTS()}
	err = txn2.LockKeys(ctx, lockCtx2, k2)
	require.NoError(t, err, "k2 lock was not cleared on child region after split rollback")

	// 4. Verify RegionCache updated to reflect the new child region
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

	/*
	 * INVARIANT ENFORCEMENT POINT:
	 * During 2PC Commit, secondary commit for k2 must reach child region R2
	 * even though client-go initially routes according to the pre-split cache.
	 */
	require.NoError(t, txn1.Commit(ctx))

	// 3. Confirm Txn2 can read the committed value on the child region
	txn2, err := store.Begin()
	require.NoError(t, err)
	defer txn2.Rollback()

	gotVal, err := txn2.Get(ctx, k2)
	require.NoError(t, err)
	require.Equal(t, val2, gotVal)
}
