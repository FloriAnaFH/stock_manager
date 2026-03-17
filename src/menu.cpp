#include "../include/menu.h"
#include "../include/batch.h"
#include "../include/hash_table.h"
#include "../include/plot.h"
#include "../include/stock.h"
#include "../include/table_stats.h"
#include "../include/tui.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

void run_menu ( HashTable &ht, owner_t &owner ) {
    std::vector<std::string> output;
    std::string status;

    while ( true ) {
        std::string line = draw_screen ( output, status );
        if ( line.empty() && std::cin.eof() )
            break;

        std::istringstream ss ( line );
        std::string cmd;
        ss >> cmd;

        // normalise to upper-case so "b" and "B" both work
        for ( auto &c : cmd )
            c = static_cast<char> ( toupper ( static_cast<unsigned char> ( c ) ) );

        output.clear();
        status.clear();

        if ( cmd == "9" || cmd == "QUIT" )
            break;

        //  BATCH IMPORT
        if ( cmd == "B" || cmd == "BATCH" ) {
            auto dir = input_screen ( output, status, "Directory of CSV files [../stock_prices/]:" );
            if ( dir.empty() )
                dir = "../stock_prices/";
            status = "Scanning " + dir + " …";

            BatchResult res = batch_import ( dir, ht, owner );

            // ── summary header ────────────────────────────────────────────
            output.push_back ( "  Batch import: " + dir );
            output.push_back ( "  ─────────────────────────────────────────" );

            auto rowB = [] ( const std::string &label, const std::string &value ) -> std::string {
                std::ostringstream o;
                o << "    " << std::left << std::setw ( 14 ) << label << value;
                return o.str();
            };

            output.push_back ( rowB ( "Imported:", std::to_string ( res.imported ) ) );
            output.push_back ( rowB ( "Skipped:", std::to_string ( res.skipped ) ) );
            output.push_back ( rowB ( "Failed:", std::to_string ( res.failed ) ) );
            output.push_back ( "" );

            // ── per-file detail ───────────────────────────────────────────
            static constexpr int MAX_OK_LINES = 8; // only display 8 lines of data that imported OK to fit window
            int okShown = 0;

            for ( const auto &e : res.entries ) {
                if ( e.ok ) {
                    if ( okShown < MAX_OK_LINES ) {
                        output.push_back ( "  \u2713 " + e.ticker + "  " + e.name + "  [" + e.wkn + "]" );
                        ++okShown;
                    } else if ( okShown == MAX_OK_LINES ) {
                        int remaining = res.imported - MAX_OK_LINES;
                        output.push_back ( "    \u2026 and " + std::to_string ( remaining ) + " more imported." );
                        ++okShown;  // only emit once
                    }
                } else {
                    std::string prefix = ( e.errMsg == "Already in table \u2013 skipped" ) ? "  ~ " : "  \u2717 ";
                    output.push_back ( prefix + e.filename + ": " + e.errMsg );
                }
            }

            status = "Batch import done  \u2013  " + std::to_string ( res.imported ) + " imported, " +
                     std::to_string ( res.skipped ) + " skipped, " + std::to_string ( res.failed ) + " failed";

            //  STATS
        } else if ( cmd == "0" || cmd == "STATS" ) {
            status = "Running diagnostics\u2026";
            TableStats s = ht.tableStats();

            auto fmtNs = [] ( double ns ) -> std::string {
                std::ostringstream o;
                if ( ns >= 1000.0 )
                    o << std::fixed << std::setprecision ( 3 ) << ns / 1000.0 << " \u00b5s";
                else
                    o << std::fixed << std::setprecision ( 2 ) << ns << " ns";
                return o.str();
            };

            auto pct = [] ( double v ) -> std::string {
                std::ostringstream o;
                o << std::fixed << std::setprecision ( 1 ) << v * 100.0 << " %";
                return o.str();
            };

            auto rowS = [] ( const std::string &label, const std::string &value ) -> std::string {
                std::ostringstream o;
                o << "    " << std::left << std::setw ( 24 ) << label << value;
                return o.str();
            };

            // ── header ────────────────────────────────────────────────────
            // draw frame for heeader and print header
            output.push_back ( "  "
                               "\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
                               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
                               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557" );
            output.push_back ( "  \u2551      Hash-Table Diagnostics          \u2551" );
            output.push_back ( "  "
                               "\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
                               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550"
                               "\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d" );
            output.push_back ( "" );

            // ── structural stats ──────────────────────────────────────────
            output.push_back ( "  Structure" );
            output.push_back ( "  "
                               "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                               "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                               "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500" );
            output.push_back ( rowS ( "Entries (live):", std::to_string ( s.entries ) ) );
            output.push_back ( rowS ( "Capacity (slots):", std::to_string ( s.capacity ) ) );
            output.push_back ( rowS ( "Tombstones:", std::to_string ( s.deleted ) ) );
            output.push_back ( rowS ( "Load factor:", pct ( s.loadFactor ) ) );
            output.push_back ( rowS ( "Max load factor:", pct ( 0.75 ) ) );
            output.push_back ( "" );

            // ── probing stats ─────────────────────────────────────────────
            output.push_back ( "  Probing" );
            output.push_back ( "  "
                               "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                               "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                               "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500" );
            {
                std::ostringstream o;
                o << std::fixed << std::setprecision ( 2 ) << s.avgProbeLen;
                output.push_back ( rowS ( "Avg probe groups:", o.str() ) );
                output.push_back ( rowS ( "Group size (SIMD):", "32 slots" ) );
            }
            output.push_back ( "" );

            // ── timing benchmark ──────────────────────────────────────────
            output.push_back ( "  Timing  (200 rounds \u00d7 all live keys)" );
            output.push_back ( "  "
                               "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                               "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500"
                               "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500" );
            output.push_back ( rowS ( "Avg lookup:", fmtNs ( s.avgLookupNs ) ) );
            output.push_back ( rowS ( "Avg insert:", fmtNs ( s.avgInsertNs ) ) );
            output.push_back ( rowS ( "Avg delete:", fmtNs ( s.avgDeleteNs ) ) );

            status = "Hash-table diagnostics  (" + std::to_string ( s.entries ) + " entries, " + pct ( s.loadFactor ) +
                     " load)";

            //  ADD
        } else if ( cmd == "1" || cmd == "ADD" ) {
            auto name = input_screen ( output, "ADD \u2013 enter stock details", "Stock name:" );
            auto wkn = input_screen ( output, "ADD \u2013 " + name, "WKN:" );
            auto symbol = input_screen ( output, "ADD \u2013 " + name, "Symbol:" );
            try {
                // create unique prt to Stock
                auto s = std::make_unique<Stock> ( name, wkn, symbol );
                ht.ht_insert ( name, s.get() );
                ht.ht_insert ( symbol, s.get() );
                owner.push_back ( std::move ( s ) );
                status = "Added: " + name + "  (" + symbol + ")";
            } catch ( const std::exception &e ) {
                status = std::string ( "Error: " ) + e.what();
            }

            //  DEL
        } else if ( cmd == "2" || cmd == "DEL" ) {
            auto key = input_screen ( output, status, "Name or symbol to delete:" );
            if ( Stock *st = ht.ht_lookup ( key ) ) {
                std::string name = st->getName();
                std::string symbol = st->getSymbol();
                ht.ht_delete ( name );
                ht.ht_delete ( symbol );
                owner.erase (
                    std::remove_if ( owner.begin(), owner.end(), [ & ] ( const auto &p ) { return p.get() == st; } ),
                    owner.end() );
                status = "Deleted: " + name + "  (" + symbol + ")";
            } else {
                status = "Not found: " + key;
            }

            //  IMPORT
        } else if ( cmd == "3" || cmd == "IMPORT" ) {
            auto key = input_screen ( output, status, "Name or symbol:" );
            auto file = input_screen ( output, "IMPORT \u2013 " + key, "CSV file path:" );
            if ( Stock *st = ht.ht_lookup ( key ) ) {
                try {
                    st->importCSV ( file );
                    status = "Imported CSV for: " + st->getName();
                } catch ( const std::exception &e ) {
                    status = std::string ( "Error: " ) + e.what();
                }
            } else {
                status = "Stock not found: " + key;
            }

            //  SEARCH
        } else if ( cmd == "4" || cmd == "SEARCH" ) {
            auto key = input_screen ( output, status, "Name or symbol:" );
            if ( const Stock *st = ht.ht_lookup ( key ) ) {
                output.push_back ( "  Name:   " + st->getName() );
                output.push_back ( "  Symbol: " + st->getSymbol() );
                output.push_back ( "  WKN:    " + st->getWKN() );
                if ( st->hasHistory() ) {
                    const auto &e = st->latest();
                    output.push_back ( "" );
                    output.push_back ( "  Latest  (" + e.date + ")" );
                    output.push_back ( "    Open:   " + std::to_string ( e.open ) );
                    output.push_back ( "    High:   " + std::to_string ( e.high ) );
                    output.push_back ( "    Low:    " + std::to_string ( e.low ) );
                    output.push_back ( "    Close:  " + std::to_string ( e.close ) );
                    output.push_back ( "    Volume: " + std::to_string ( e.volume ) );
                } else {
                    output.push_back ( "" );
                    output.push_back ( "  No price history available." );
                }
                status = "Search result for: " + key;
            } else {
                status = "Not found: " + key;
            }

            //  PLOT
        } else if ( cmd == "5" || cmd == "PLOT" ) {
            auto key = input_screen ( output, status, "Name or symbol:" );
            if ( const Stock *st = ht.ht_lookup ( key ) ) {
                output = plotStock ( *st );
                status = "Plot: " + st->getName();
            } else {
                status = "Not found: " + key;
            }

            //  LIST
        } else if ( cmd == "6" || cmd == "LIST" ) {
            output = ht.listAll();
            status = std::to_string ( ht.size() ) + " stock(s) in table";

            //  SAVE
        } else if ( cmd == "7" || cmd == "SAVE" ) {
            auto file = input_screen ( output, status, "Save to file:" );
            try {
                ht.save ( file );
                status = "Saved " + std::to_string ( ht.size() ) + " stock(s) to " + file;
            } catch ( const std::exception &e ) {
                status = std::string ( "Error: " ) + e.what();
            }

            //  LOAD
        } else if ( cmd == "8" || cmd == "LOAD" ) {
            auto file = input_screen ( output, status, "Load from file:" );
            try {
                auto stocks = ht.load ( file );
                std::size_t n = stocks.size();
                for ( auto &s : stocks ) {
                    ht.ht_insert ( s->getName(), s.get() );
                    ht.ht_insert ( s->getSymbol(), s.get() );
                    owner.push_back ( std::move ( s ) );
                }
                status = "Loaded " + std::to_string ( n ) + " stock(s) from " + file;
            } catch ( const std::exception &e ) {
                status = std::string ( "Error: " ) + e.what();
            }

            //  unknown
        } else if ( !cmd.empty() ) {
            status = "Unknown command: " + cmd;
        }
    }
}
