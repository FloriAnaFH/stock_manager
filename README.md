# Stock Manager

a simple cli stock manager written in c++

## build with make:

```bash
# build
make

# run with:
./stock_manager

# get help:
make help

# make options:
make clean          # remove build files
make release        # build with optimizations and no debug info
make test           # run tests
make avx2           # build with avx2 optimizations
make avx2-release   # build with avx2 optimizations and no debug info

```

avx2 requires a cpu and compiler support. If you get an error about avx2 not being supported, try building without it.

## Menu options:

0 STATS - show hash table stats and 'mini' benchmarks ( 200 runs )
1 ADD - add a new stock ( name, symbol, wkn )
2 DELETE - delete a stock by name or symbol
3 IMPORT - import historical prices for a given stock
4 SEARCH - search for a stock by name or symbol
5 PLOT - plot the last 30 days of historical prices for a given stock
6 LIST - list all stocks in the database
7 SAVE - save the database to a file
8 LOAD - load the database from a file
9 QUIT

Enter number or string ( 0-9 or command name ):
