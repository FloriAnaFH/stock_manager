## Model:

- Swiss Table Style with SIMD probing and Robin Hood insert algorithm
## Swiss Table

#### Complexity:
	- O(|key|) → key ⇐ 30 chars: O(1)

#### Description:
- split an 8-byte-hash into one byte for metadata and 7 byte hash
	- metadata byte stores:
		- state ( EMPTY, DELETED, OCCUPIED )
		- fingerprint 
- advantage:
	- performance gain by comparing fingerprints first and only comparing keys ( CPU heavier ) once fp match has been found.

## Robin Hood:

#### Concept:
  

#### Complexity:
	O(1)

- if calculated index is already taken, use slot for insert and move previous entry to next free slot

- advantage:
	- increase insert speed 

## Chosen Hash-Algorthm

####  FNV-1a ( Foller-Noll-Vo variant 1a)

	FNV-1a offset basis: 14695981039346656037ULL
	FNV-1a prime:        1099511628211ULL

- non cryptographic hash
- converts data ( string, integers, .. ) into fixed-size integer hash
	- XOR hash with next byte
	- multiply by special prime number
	
		- avalanche effect:
			- lower bits affect higher bits
			- higher bits feed back into lower bits
			- carries create nonlinear mixing - small changes flip many bits
			- → **one bit change in input affects many bits in state**
		
```cpp
uint64_t HashTable::hashString ( const std::string &key ) noexcept {
    uint64_t hash = 14695981039346656037ULL;  // FNV-1a offset basis
    for ( char c : key ) {
        hash ^= static_cast<uint8_t> ( c );
        hash *= 1099511628211ULL;  // FNV-1a prime
    }
    return hash;
}
```

#### How FNV primes are chosen:

- large Hamming weight ( number of bits set to 1 ) - high Hamming weight → better mixing
- structured to spread bits efficiently
- multiplication is fast on CPUs
- avoid clustering

#### Why FNV-1a:

 - fast and lightweight
 - produces a well-distributed integer hash
- 

## SIMD:

"Single Instruction Multiple Data"

- one cache line = 32 bytes → load 32 values to cache and apply the same operation to each simultaneously 
- does not improve or change the algorithm but improves performance by reducing the number of CPU cycles needed for comparing a number of keys ( SIMD - 32 comparisons in one CPU cycle, without: 32 CPU cycles! )
- benefit from the fact, that CPU always loads a full cache line → use the data already in cache instead of constantly loading from memory to cache ( slow )

## Overall Complexity
