# Stock Manager

a simple cli stock manager written in c++ that uses a high-performance Swiss-Table-style hash table that uses AVX2 to manage stock data.

## Design Features:

- open adressing with metadata byte
- tomnstone reuse
- (optional) AVX2 vector search ( compare 32 bytes simultaniously )
- plot historical price data ( ASCI Plot )

## Build dependencies:

```bash
g++ ( with c++17 support )
make
```

## Build and Run

```bash
# clone the repository
git clone https://github.com/floegger/stock_manager.git

# build: make [options]
make                # build with default options

# options:
clean          # remove build files
release        # build with optimizations and no debug info
test           # run tests
avx2           # build with avx2 optimizations*
avx2-release   # build with avx2 optimizations and no debug info

# get help:
make help

# run:
cd build # binaries  are located in the build/ folder
./stock_manager
```

    *avx2 requires a cpu and compiler support. If you get an error about avx2 not being supported, try building without it.

## Menu options:

0 STATS - show hash table stats and 'mini' benchmarks ( 200 runs ) <br/>
1 ADD - add a new stock ( name, symbol, wkn ) <br/>
2 DELETE - delete a stock by name or symbol<br/>
3 IMPORT - import historical prices for a given stock<br/>
4 SEARCH - search for a stock by name or symbol<br/>
5 PLOT - plot the last 30 days of historical prices for a given stock<br/>
6 LIST - list all stocks in the database<br/>
7 SAVE - save the database to a file<br/>
8 LOAD - load the database from a file<br/>
9 QUIT<br/>
B BATCH - batch import stock prices from a folder<br/>

Enter number or string ( 0-9 or command name ):
