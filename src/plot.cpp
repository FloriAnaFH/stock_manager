#include "../include/plot.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Constants that must match the TUI inner width (62).
// Y-axis label + separator eats YLABEL_W columns, leaving PLOT_W for data.
// ---------------------------------------------------------------------------
static constexpr int INNER_W  = 62;
static constexpr int YLABEL_W = 10;   // "  9999.99 |"  → 10 chars
static constexpr int PLOT_W   = INNER_W - YLABEL_W;   // 52 columns for data
static constexpr int PLOT_ROWS = 14;   // number of price rows

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static std::string priceLabel ( double v ) {
    char buf[ 16 ];
    std::snprintf ( buf, sizeof ( buf ), "%8.2f |", v );
    return std::string ( buf );   // always 10 chars
}

// Return the date string trimmed to at most maxLen characters.
static std::string trimDate ( const std::string &d, int maxLen ) {
    if ( static_cast<int> ( d.size() ) <= maxLen )
        return d;
    return d.substr ( 0, static_cast<std::size_t> ( maxLen ) );
}

// ---------------------------------------------------------------------------
// plotStock
// ---------------------------------------------------------------------------
std::vector<std::string> plotStock ( const Stock &stock ) {
    std::vector<std::string> lines;
    const auto &history = stock.history();

    if ( history.empty() ) {
        lines.push_back ( "  No price data for: " + stock.getName() );
        return lines;
    }

    // ── 1. gather close prices & map each onto a column 0..PLOT_W-1 ────────
    const int N        = static_cast<int> ( history.size() );
    const int cols     = std::min ( N, PLOT_W );

    // Sample N data points down to `cols` columns by picking the closest
    // index for each column.
    std::vector<double> sampled ( static_cast<std::size_t> ( cols ) );
    std::vector<int>    srcIdx  ( static_cast<std::size_t> ( cols ) );
    for ( int c = 0; c < cols; ++c ) {
        int idx = ( N == 1 ) ? 0
                             : static_cast<int> ( std::round (
                                   c * ( N - 1.0 ) / ( cols - 1.0 ) ) );
        idx          = std::clamp ( idx, 0, N - 1 );
        srcIdx[ static_cast<std::size_t> ( c ) ]  = idx;
        sampled[ static_cast<std::size_t> ( c ) ] = history[ static_cast<std::size_t> ( idx ) ].close;
    }

    // ── 2. price range ───────────────────────────────────────────────────────
    double lo = *std::min_element ( sampled.begin(), sampled.end() );
    double hi = *std::max_element ( sampled.begin(), sampled.end() );
    if ( hi <= lo ) hi = lo + 1.0;

    // ── 3. title ─────────────────────────────────────────────────────────────
    {
        std::string title = "  " + stock.getName() + " (" + stock.getSymbol() + ")"
                            + "  –  close price,  " + std::to_string ( N ) + " sessions";
        if ( static_cast<int> ( title.size() ) > INNER_W )
            title = title.substr ( 0, static_cast<std::size_t> ( INNER_W ) );
        lines.push_back ( title );
        lines.push_back ( "" );
    }

    // ── 4. build the 2-D grid (PLOT_H rows × cols columns) ───────────────────
    //   grid[row][col] is true when that cell should be filled.
    //   row 0 = top (highest price), row PLOT_H-1 = bottom (lowest price).
    std::vector<std::vector<char>> grid (
        static_cast<std::size_t> ( PLOT_ROWS ),
        std::vector<char> ( static_cast<std::size_t> ( cols ), ' ' ) );

    // For each column, compute the row that the price falls on.
    std::vector<int> priceRow ( static_cast<std::size_t> ( cols ) );
    for ( int c = 0; c < cols; ++c ) {
        double norm = ( sampled[ static_cast<std::size_t> ( c ) ] - lo ) / ( hi - lo );  // 0..1
        int    row  = static_cast<int> ( std::round ( ( 1.0 - norm ) * ( PLOT_ROWS - 1 ) ) );
        row = std::clamp ( row, 0, PLOT_ROWS - 1 );
        priceRow[ static_cast<std::size_t> ( c ) ] = row;
    }

    // Place the dot for each column, and draw a vertical bar connecting
    // adjacent points so there are no gaps.
    for ( int c = 0; c < cols; ++c ) {
        int r = priceRow[ static_cast<std::size_t> ( c ) ];
        grid[ static_cast<std::size_t> ( r ) ][ static_cast<std::size_t> ( c ) ] = '*';

        if ( c + 1 < cols ) {
            int r2    = priceRow[ static_cast<std::size_t> ( c + 1 ) ];
            int rLo   = std::min ( r, r2 );
            int rHi   = std::max ( r, r2 );
            for ( int fill = rLo; fill <= rHi; ++fill )
                if ( grid[ static_cast<std::size_t> ( fill ) ][ static_cast<std::size_t> ( c ) ] != '*' )
                    grid[ static_cast<std::size_t> ( fill ) ][ static_cast<std::size_t> ( c ) ] = '|';
        }
    }

    // ── 5. render price rows ─────────────────────────────────────────────────
    for ( int r = 0; r < PLOT_ROWS; ++r ) {
        // Y-axis label only on rows that are multiples of (PLOT_ROWS-1)/3
        // so we get ~4 evenly spaced tick marks (top, 2/3, 1/3, bottom).
        bool showTick = ( r == 0 ) || ( r == PLOT_ROWS - 1 )
                        || ( r == ( PLOT_ROWS - 1 ) / 3 )
                        || ( r == 2 * ( PLOT_ROWS - 1 ) / 3 );
        double tickPrice = hi - r * ( hi - lo ) / ( PLOT_ROWS - 1 );

        std::string line;
        if ( showTick ) {
            line = priceLabel ( tickPrice );
        } else {
            line = std::string ( static_cast<std::size_t> ( YLABEL_W - 1 ), ' ' ) + "|";
        }

        for ( int c = 0; c < cols; ++c )
            line += grid[ static_cast<std::size_t> ( r ) ][ static_cast<std::size_t> ( c ) ];

        lines.push_back ( line );
    }

    // ── 6. x-axis rule ───────────────────────────────────────────────────────
    {
        std::string rule ( static_cast<std::size_t> ( YLABEL_W - 1 ), ' ' );
        rule += '+';
        rule += std::string ( static_cast<std::size_t> ( cols ), '-' );
        lines.push_back ( rule );
    }

    // ── 7. date labels on x-axis ─────────────────────────────────────────────
    // Pick ~4 evenly-spaced tick positions and render their dates.
    // Dates are printed left-aligned from the tick column; we clip if they
    // would overlap with the next one or run off the right edge.
    {
        // Number of date ticks we want (including first and last).
        static constexpr int XTICKS = 4;
        // date labels row: starts with YLABEL_W spaces, then cols characters
        std::string dateLine ( static_cast<std::size_t> ( YLABEL_W + cols ), ' ' );

        for ( int t = 0; t < XTICKS; ++t ) {
            int col = ( XTICKS == 1 ) ? 0
                                      : static_cast<int> ( std::round (
                                            t * ( cols - 1.0 ) / ( XTICKS - 1.0 ) ) );
            col = std::clamp ( col, 0, cols - 1 );

            const std::string &date = history[ static_cast<std::size_t> ( srcIdx[ static_cast<std::size_t> ( col ) ] ) ].date;

            // How many characters can we safely write starting at this col?
            int nextTick = ( t + 1 < XTICKS )
                               ? static_cast<int> ( std::round (
                                     ( t + 1 ) * ( cols - 1.0 ) / ( XTICKS - 1.0 ) ) )
                               : cols;
            int maxLen = nextTick - col - 1;   // leave 1-space gap
            if ( maxLen < 1 ) maxLen = 1;

            std::string label = trimDate ( date, maxLen );
            int pos = YLABEL_W + col;
            for ( int i = 0; i < static_cast<int> ( label.size() ); ++i )
                dateLine[ static_cast<std::size_t> ( pos + i ) ] = label[ static_cast<std::size_t> ( i ) ];
        }

        // trim trailing spaces
        auto last = dateLine.find_last_not_of ( ' ' );
        if ( last != std::string::npos )
            dateLine = dateLine.substr ( 0, last + 1 );

        lines.push_back ( dateLine );
    }

    return lines;
}
