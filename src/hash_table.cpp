#include "../include/hash_table.h"

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <set>
#include <sstream>

#ifdef __AVX2__
#include <immintrin.h>
#endif

/* performance enhancements:
 *  - replace '%capacity' with '& mask_' 20-80 CPU cycles vs. 1 cycle
 */
HashTable::HashTable ( std::size_t capacity )
    : ctrl_ ( pow2 ( capacity ), EMPTY ), entries_ ( pow2 ( capacity ) ), size_ ( 0 ), mask_ ( pow2 ( capacity ) - 1 ) {
    if ( capacity == 0 ) {
        throw std::invalid_argument ( "Capacity must be > 0" );
    }
}  // HashTable

/* === Hashing and fingerprinting === */

uint64_t HashTable::hashString ( const std::string &key ) noexcept {
    uint64_t hash = 14695981039346656037ULL;  // FNV-1a offset basis
    for ( char c : key ) {
        hash ^= static_cast<uint8_t> ( c );
        hash *= 1099511628211ULL;  // FNV-1a prime
    }
    return hash;
}

uint8_t HashTable::fingerprint ( uint64_t hash ) noexcept {
    return static_cast<uint8_t> ( hash & 0x7F );  // Use lower 7 bits for fingerprint
}

/* === Probe function === */

// insert = true -> insert, insert = false -> lookup
std::ptrdiff_t HashTable::probe ( const std::string &key, uint64_t hash, bool insert ) const noexcept {

    const uint8_t fp = fingerprint ( hash );

#ifdef __AVX2__
    std::size_t group = ( hash & mask_ ) & ~( GROUP_SIZE - 1 );  // Align to group boundary
    __m256i fpVec = _mm256_set1_epi8 ( fp );
    __m256i emptyVec = _mm256_set1_epi8 ( static_cast<char> ( EMPTY ) );

    std::ptrdiff_t firstTombstone = -1;

    for ( ;; ) {
        __m256i ctrlVec = _mm256_loadu_si256 ( reinterpret_cast<const __m256i *> ( &ctrl_[ group ] ) );
        uint32_t matchMask = _mm256_movemask_epi8 ( _mm256_cmpeq_epi8 ( ctrlVec, fpVec ) );
        uint32_t emptyMask = _mm256_movemask_epi8 ( _mm256_cmpeq_epi8 ( ctrlVec, emptyVec ) );

        // check candidates
        for ( uint32_t mask = matchMask; mask != 0; mask &= ( mask - 1 ) ) {
            std::ptrdiff_t idx = ( group + static_cast<std::size_t> ( __builtin_ctz ( mask ) ) ) & mask_;

            if ( entries_[ idx ].hash == hash && entries_[ idx ].key == key ) {
                return static_cast<std::ptrdiff_t> ( idx );  // Found
            }
        }
        // record first tombstone for this group
        if ( insert && firstTombstone == -1 ) {
            for ( std::size_t i = 0; i < GROUP_SIZE; ++i ) {
                std::size_t idx = ( group + i ) & mask_;
                if ( ctrl_[ idx ] == DELETED ) {
                    firstTombstone = static_cast<std::ptrdiff_t> ( idx );
                    break;
                }
            }
        }

        if ( emptyMask ) {
            if ( !insert )
                return -1;  // Not found
            std::size_t slot = ( group + static_cast<std::size_t> ( __builtin_ctz ( emptyMask ) ) ) & mask_;
            return firstTombstone != -1 ? firstTombstone : static_cast<std::ptrdiff_t> ( slot );  // Insert here
        }
        group = ( group + GROUP_SIZE ) & mask_;  // Move to next group
    }
#else  // Fallback to scalar probing
    std::size_t group = ( hash & mask_ ) & ~( GROUP_SIZE - 1 );  // Align to group boundary
    std::ptrdiff_t firstTombstone = -1;

    for ( ;; ) {
        bool foundEmpty = false;
        std::size_t emptyIdx = 0;

        for ( std::size_t i = 0; i < GROUP_SIZE; ++i ) {
            std::size_t idx = ( group + i ) & mask_;
            uint8_t c = ctrl_[ idx ];

            if ( c == EMPTY ) {
                if ( !foundEmpty ) {
                    foundEmpty = true;
                    emptyIdx = idx;
                }
            } else if ( c == DELETED ) {
                if ( insert && firstTombstone == -1 ) {
                    firstTombstone = static_cast<std::ptrdiff_t> ( idx );
                }
            } else if ( c == fp && entries_[ idx ].hash == hash && entries_[ idx ].key == key ) {
                return static_cast<std::ptrdiff_t> ( idx );  // Found
            }
        }

        if ( foundEmpty ) {
            if ( !insert )
                return -1;                                                                            // Not found
            return firstTombstone != -1 ? firstTombstone : static_cast<std::ptrdiff_t> ( emptyIdx );  // Insert here
        }
        group = ( group + GROUP_SIZE ) & mask_;  // Move to next group
    }  // end probeloop
#endif
}  // probe

/* === Insert, lookup, delete === */

void HashTable::ht_insert ( const std::string &key, Stock *stock ) {
    if ( static_cast<double> ( size_ ) / static_cast<double> ( ctrl_.size() ) > MAX_LOAD )
        rehash ( ctrl_.size() * 2 );
    uint64_t hash = hashString ( key );
    auto idx = static_cast<std::size_t> ( probe ( key, hash, true ) );
    if ( ctrl_[ idx ] == EMPTY || ctrl_[ idx ] == DELETED ) {
        if ( ctrl_[ idx ] == DELETED )
            --deleted_;  // Reusing a tombstone slot, decrement counter
        Entry tmp;
        tmp.key = key;
        tmp.hash = hash;
        tmp.stock = stock;
        entries_[ idx ] = std::move ( tmp );
        ++size_;
    } else {
        entries_[ idx ].stock = stock;  // Update existing
    }
    ctrl_[ idx ] = fingerprint ( hash );
}  // ht_insert

/* === lookup === */

Stock *HashTable::ht_lookup ( const std::string &key ) const {
    uint64_t hash = hashString ( key );
    auto idx = probe ( key, hash, false );
    if ( idx != -1 ) {
        return entries_[ static_cast<std::size_t> ( idx ) ].stock;
    }
    return nullptr;  // Not found
}  // ht_lookup

/* === delete === */

bool HashTable::ht_delete ( const std::string &key ) {
    uint64_t hash = hashString ( key );
    auto raw = probe ( key, hash, false );

    if ( raw == -1 )
        return false;  // Not found

    auto idx = static_cast<std::size_t> ( raw );
    ctrl_[ idx ] = DELETED;
    entries_[ idx ] = Entry();  // Clear entry
    --size_;
    ++deleted_;  // track tombstones for rehash

    // rehash when ratio tombstone/capacity > 20%
    if ( static_cast<double> ( deleted_ ) / static_cast<double> ( ctrl_.size() ) > MAX_TOMB ) {
        rehash ( ctrl_.size() );
    }

    return true;
}  // ht_delete

/* === Rehashing === */

void HashTable::rehash ( std::size_t newCapacity ) {
    HashTable newTable ( newCapacity );
    for ( std::size_t i = 0; i < ctrl_.size(); ++i ) {
        if ( ctrl_[ i ] != EMPTY && ctrl_[ i ] != DELETED ) {
            const Entry &entry = entries_[ i ];
            newTable.ht_insert ( entry.key, entry.stock );
        }
    }
    ctrl_ = std::move ( newTable.ctrl_ );
    entries_ = std::move ( newTable.entries_ );
    size_ = newTable.size_;
    mask_ = ctrl_.size() - 1;  // Update mask for new capacity
    deleted_ = 0;              // reset tombstone counter
}  // rehash

// === List all entries ===
std::vector<std::string> HashTable::listAll () const {
    // ── collect unique Stock pointers ────────────────────────────────────────
    std::set<const Stock *> seen;
    std::vector<const Stock *> stocks;
    for ( std::size_t i = 0; i < ctrl_.size(); ++i ) {
        if ( ctrl_[ i ] != EMPTY && ctrl_[ i ] != DELETED ) {
            const Stock *s = entries_[ i ].stock;
            if ( seen.insert ( s ).second )
                stocks.push_back ( s );
        }
    }

    if ( stocks.empty() ) {
        return { "  (no stocks)" };
    }

    //  sort alphabetically by name
    std::sort ( stocks.begin(), stocks.end(),
                [] ( const Stock *a, const Stock *b ) { return a->getName() < b->getName(); } );

    static constexpr int W_NAME = 22;
    static constexpr int W_SYM = 6;
    static constexpr int W_WKN = 8;
    static constexpr int W_CLOSE = 9;
    static constexpr int W_DATE = 10;

    auto clip = [] ( const std::string &s, int w ) -> std::string {
        if ( static_cast<int> ( s.size() ) <= w )
            return s;
        return s.substr ( 0, static_cast<std::size_t> ( w - 1 ) ) + "\xe2\x80\xa6";  // …
    };

    std::vector<std::string> lines;

    //  header
    {
        std::ostringstream h;
        h << "  " << std::left << std::setw ( W_NAME ) << "Name"
          << "  " << std::left << std::setw ( W_SYM ) << "Sym"
          << " " << std::left << std::setw ( W_WKN ) << "WKN"
          << " " << std::right << std::setw ( W_CLOSE ) << "Close"
          << "  " << std::left << "Date";
        lines.push_back ( h.str() );
        // divider
        lines.push_back ( "  " + std::string ( W_NAME, '-' ) + "  " + std::string ( W_SYM, '-' ) + " " +
                          std::string ( W_WKN, '-' ) + " " + std::string ( W_CLOSE, '-' ) + "  " +
                          std::string ( W_DATE, '-' ) );
    }

    //  rows
    for ( const Stock *s : stocks ) {
        std::ostringstream row;

        // name
        row << "  " << std::left << std::setw ( W_NAME ) << clip ( s->getName(), W_NAME );

        // symbol
        row << "  " << std::left << std::setw ( W_SYM ) << clip ( s->getSymbol(), W_SYM );

        // WKN
        row << " " << std::left << std::setw ( W_WKN ) << clip ( s->getWKN(), W_WKN );

        // close + date
        if ( s->hasHistory() ) {
            const PriceEntry &e = s->latest();
            row << " " << std::right << std::fixed << std::setprecision ( 2 ) << std::setw ( W_CLOSE ) << e.close;
            row << "  " << std::left << e.date;
        } else {
            row << " " << std::right << std::setw ( W_CLOSE ) << "—"
                << "  " << std::left << "no data";
        }

        lines.push_back ( row.str() );
    }

    //  footer
    lines.push_back ( "" );
    lines.push_back ( "  " + std::to_string ( stocks.size() ) + " stock(s)" );

    return lines;
}  // listAll

/* === Save and load === */

void HashTable::save ( const std::string &filename ) const {
    std::ofstream file ( filename, std::ios::binary );
    if ( !file )
        throw std::runtime_error ( "Cannot write: " + filename );

    file << size_ << "\n";  // Save number of entries

    for ( std::size_t i = 0; i < ctrl_.size(); ++i ) {
        if ( ctrl_[ i ] != EMPTY && ctrl_[ i ] != DELETED ) {
            entries_[ i ].stock->saveToFile ( file );
        }
    }

}  // save

std::vector<std::unique_ptr<Stock>> HashTable::load ( const std::string &filename ) {
    std::ifstream file ( filename );
    if ( !file )
        throw std::runtime_error ( "Cannot read: " + filename );
    std::fill ( ctrl_.begin(), ctrl_.end(), EMPTY );  // Clear existing data
    for ( auto &entry : entries_ ) {
        entry = Entry {};  // Clear entries
    }
    size_ = 0;
    std::size_t n = 0;
    file >> n;  // Read number of entries
    file.ignore();

    std::vector<std::unique_ptr<Stock>> result;
    result.reserve ( n );
    for ( std::size_t i = 0; i < n; ++i ) {
        result.push_back ( std::make_unique<Stock> ( Stock::loadFromFile ( file ) ) );
    }

    return result;
}

std::size_t HashTable::pow2 ( std::size_t n ) {
    if ( n == 0 )
        return 1;
    --n;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;

    return n + 1;
}
