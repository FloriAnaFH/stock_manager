#ifndef STOCK_H
#define STOCK_H

#include "priceEntry.h"

#include <stdexcept>
#include <string>
#include <vector>

class Stock {
  public:
    Stock ( std::string name, std::string wkn, std::string symbol );

    const std::string &getName () const noexcept { return name_; }
    const std::string &getWKN () const noexcept { return wkn_; }
    const std::string &getSymbol () const noexcept { return symbol_; }
    bool hasHistory () const noexcept { return !history_.empty(); }
    const PriceEntry &latest () const;
    const std::vector<PriceEntry> &history () const noexcept { return history_; }  // returns ref to history vector

    void importCSV ( const std::string &filename );
    void saveToFile ( std::ostream &outputStram ) const;
    static Stock loadFromFile ( std::istream &inputStram );

  private:
    static PriceEntry parseCSVLine(const std::string &line);
    std::string name_, wkn_, symbol_;
    std::vector<PriceEntry> history_;

};  // class Stock
#endif
