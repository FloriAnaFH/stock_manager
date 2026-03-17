#include "../include/table_stats.h"
#include "../include/hash_table.h"

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

TableStats HashTable::tableStats ( int benchRounds ) const {
    TableStats stats;
    stats.entries = size_;
    stats.capacity = ctrl_.size();

    std::size_t deletedCount = 0;
    double totalProbeLen = 0.0;

    for ( std::size_t i = 0; i < ctrl_.size(); ++i ) {
        if ( ctrl_[ i ] == DELETED ) {
            ++deletedCount;
            continue;
        }
        if ( ctrl_[ i ] == EMPTY )
            continue;

        // Walk probe sequence from the ideal slot to this slot to measure
        // how many steps were needed to place / find this key.
        const uint64_t hash = entries_[ i ].hash;
        const std::size_t cap = ctrl_.size();
        std::size_t ideal = ( hash % cap ) & ~( GROUP_SIZE - 1 );
        std::size_t steps = 1;

        // Advance groups until we reach the group that contains slot i.
        std::size_t group = ideal;
        while ( true ) {
            bool inGroup = false;
            for ( std::size_t g = 0; g < GROUP_SIZE; ++g ) {
                if ( ( group + g ) % cap == i ) {
                    inGroup = true;
                    break;
                }
            }
            if ( inGroup )
                break;
            group = ( group + GROUP_SIZE ) % cap;
            ++steps;
        }
        totalProbeLen += static_cast<double> ( steps );
    }

    stats.deleted = deletedCount;
    stats.loadFactor =
        stats.capacity > 0 ? static_cast<double> ( stats.entries ) / static_cast<double> ( stats.capacity ) : 0.0;
    stats.avgProbeLen = stats.entries > 0 ? totalProbeLen / static_cast<double> ( stats.entries ) : 0.0;

    // ── micro-benchmark ────────────────────────────────────────────────────
    // Collect all live keys once so the benchmark loop is deterministic.
    std::vector<std::string> keys;
    keys.reserve ( size_ );
    for ( std::size_t i = 0; i < ctrl_.size(); ++i ) {
        if ( ctrl_[ i ] != EMPTY && ctrl_[ i ] != DELETED )
            keys.push_back ( entries_[ i ].key );
    }

    if ( keys.empty() || benchRounds <= 0 ) {
        return stats;
    }

    // Clone into a fresh mutable table so the benchmark doesn't mutate *this.
    HashTable bench ( ctrl_.size() );
    for ( std::size_t i = 0; i < ctrl_.size(); ++i ) {
        if ( ctrl_[ i ] != EMPTY && ctrl_[ i ] != DELETED )
            bench.ht_insert ( entries_[ i ].key, entries_[ i ].stock );
    }

    const long long totalOps = static_cast<long long> ( keys.size() ) * benchRounds;

    // ── lookup timing ──────────────────────────────────────────────────────
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for ( int r = 0; r < benchRounds; ++r )
            for ( const auto &k : keys )
                ( void )bench.ht_lookup ( k );
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = static_cast<double> ( std::chrono::duration_cast<std::chrono::nanoseconds> ( t1 - t0 ).count() );
        stats.avgLookupNs = ns / static_cast<double> ( totalOps );
    }

    // ── insert timing (re-inserting existing keys = update path) ──────────
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        for ( int r = 0; r < benchRounds; ++r )
            for ( std::size_t ki = 0; ki < keys.size(); ++ki )
                bench.ht_insert ( keys[ ki ], entries_[ static_cast<std::size_t> ( bench.probe (
                                                            keys[ ki ], bench.hashString ( keys[ ki ] ), false ) ) ]
                                                  .stock );
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = static_cast<double> ( std::chrono::duration_cast<std::chrono::nanoseconds> ( t1 - t0 ).count() );
        stats.avgInsertNs = ns / static_cast<double> ( totalOps );
    }

    // ── delete timing ──────────────────────────────────────────────────────
    // Delete then immediately re-insert each key each round to keep the
    // table intact.
    {
        long long deleteOps = 0;
        auto t0 = std::chrono::high_resolution_clock::now();
        for ( int r = 0; r < benchRounds; ++r ) {
            for ( const auto &k : keys ) {
                Stock *ptr = bench.ht_lookup ( k );
                bench.ht_delete ( k );
                bench.ht_insert ( k, ptr );
                ++deleteOps;
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns = static_cast<double> ( std::chrono::duration_cast<std::chrono::nanoseconds> ( t1 - t0 ).count() );
        // divide by 2: each iteration is one delete + one re-insert; charge only the delete
        stats.avgDeleteNs = ns / ( 2.0 * static_cast<double> ( deleteOps ) );
    }

    return stats;
}  // tableStats
