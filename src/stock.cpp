#include "../include/stock.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

Stock::Stock ( std::string name, std::string wkn, std::string symbol )
    : name_ ( std::move ( name ) ), wkn_ ( std::move ( wkn ) ), symbol_ ( std::move ( symbol ) ) {
    if ( name_.empty() || wkn_.empty() || symbol_.empty() ) {
        throw std::invalid_argument ( "Name, WKN and Symbol must not be empty." );
    }
}  // Stock

const PriceEntry &Stock::latest () const {
    if ( history_.empty() ) {
        throw std::runtime_error ( "No price history available." );
    }
    return history_.back();
}  // latest

PriceEntry Stock::parseCSVLine( const std::string &line) {

    auto stripDollar = [](std::string &s) {
        if (!s.empty() && s.front() == '$')
            s.erase(0, 1);
    };

    std::istringstream ss(line);
    PriceEntry entry;
    std::string tok;

    std::getline(ss, entry.date, ',');

    std::getline(ss, tok, ',');
    stripDollar(tok);
    entry.close  = std::stod(tok);

    std::getline(ss, tok, ',');
    entry.volume = std::stol(tok);

    std::getline(ss, tok, ',');
    stripDollar(tok);
    entry.open   = std::stod(tok);

    std::getline(ss, tok, ',');
    stripDollar(tok);
    entry.high   = std::stod(tok);

    std::getline(ss, tok, ',');
    stripDollar(tok);
    entry.low    = std::stod(tok);

    return entry;
}


void Stock::importCSV ( const std::string &filename ) {
    std::ifstream file ( filename );
    if ( !file ) {
        throw std::runtime_error ( "Could not open file: " + filename );
    }

    history_.clear();
    std::string line;
    std::getline ( file, line );  // Skip header
                                  //

    while (std::getline(file, line) && history_.size() < 30){
        history_.push_back(parseCSVLine(line));
    }
}  // importCSV

/* === Save and load === */

void Stock::saveToFile ( std::ostream &out ) const {
    out << name_ << '\n' << wkn_ << '\n' << symbol_ << '\n';
    out << history_.size() << '\n';

    for ( const auto &entry : history_ ) {
        out << entry.date << ',' << entry.close << ',' << entry.volume << ',' << entry.open << ',' << entry.high << ','
            << entry.low << '\n';
    }
}  // saveToFile

Stock Stock::loadFromFile ( std::istream &in ) {
    std::string name, wkn, symbol;
    std::getline ( in, name );
    std::getline ( in, wkn );
    std::getline ( in, symbol );

    Stock stock ( name, wkn, symbol );

    size_t n = 0;
    in >> n;
    in.ignore();  // Ignore newline after size
    stock.history_.reserve ( n );

    for (size_t i = 0; i < n; ++i) {
        std::string line;
        std::getline(in, line);
        stock.history_.push_back(parseCSVLine(line));
    }

    return stock;
}  // loadFromFile
