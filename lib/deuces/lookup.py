import itertools
from .card import Card

class LookupTable(object):
    """
    Number of Distinct Hand Values:

    Straight Flush             10
    Four of a Kind             156      [(13 choose 2) * (2 choose 1)]
    Full Houses                156      [(13 choose 2) * (2 choose 1)]
    Flush                      1277     [(13 choose 5) - 10 straight flushes]
    Straight                   10
    Three of a Kind            858      [(13 choose 3) * (3 choose 1)]
    Three Card Three of a Kind 13       [(13 choose 1)]
    Two Pair                   858      [(13 choose 3) * (3 choose 2)]
    One Pair                   2860     [(13 choose 4) * (4 choose 1)]
    Three Card One Pair        156      [(13 choose 1) * (12 choose 1)]
    High Card                  1277     [(13 choose 5) - 10 straights]
    Three Card High Card +     286      [(13 choose 3)]
    ----------------------------
    TOTAL                      7917
    """
    MAX_STRAIGHT_FLUSH    = 10
    MAX_FOUR_OF_A_KIND    = 166
    MAX_FULL_HOUSE        = 322
    MAX_FLUSH             = 1599
    MAX_STRAIGHT          = 1609
    # MAX_THREE_OF_A_KIND = 2467
    MAX_THREE_OF_A_KIND   = 2480
    # MAX_TWO_PAIR        = 3325
    MAX_TWO_PAIR          = 3338
    # MAX_PAIR            = 6185
    MAX_PAIR              = 6354
    # MAX_HIGH_CARD       = 7462
    MAX_HIGH_CARD         = 7917

    MAX_TO_RANK_CLASS = {
        MAX_STRAIGHT_FLUSH: 1,
        MAX_FOUR_OF_A_KIND: 2,
        MAX_FULL_HOUSE: 3,
        MAX_FLUSH: 4,
        MAX_STRAIGHT: 5,
        MAX_THREE_OF_A_KIND: 6,
        MAX_TWO_PAIR: 7,
        MAX_PAIR: 8,
        MAX_HIGH_CARD: 9,
    }

    RANK_CLASS_TO_STRING = {
        1 : "Straight Flush",
        2 : "Four of a Kind",
        3 : "Full House",
        4 : "Flush",
        5 : "Straight",
        6 : "Three of a Kind",
        7 : "Two Pair",
        8 : "Pair",
        9 : "High Card"
    }

    def __init__(self):
        """
        Calculates lookup tables
        """
        # create dictionaries
        self.flush_lookup = {}
        self.unsuited_lookup = {}

        # create the lookup table in piecewise fashion
        self.flushes()  # this will call straights and high cards method,
                        # we reuse some of the bit sequences
        self.multiples()

    def flushes(self):
        """
        Straight flushes and flushes. 

        Lookup is done on 13 bit integer (2^13 > 7462):
        xxxbbbbb bbbbbbbb => integer hand index
        """

        # straight flushes in rank order
        straight_flushes = [
            7936, # int('0b1111100000000', 2), # royal flush
            3968, # int('0b111110000000', 2),
            1984, # int('0b11111000000', 2),
            992, # int('0b1111100000', 2),
            496, # int('0b111110000', 2),
            248, # int('0b11111000', 2),
            124, # int('0b1111100', 2),
            62, # int('0b111110', 2),
            31, # int('0b11111', 2),
            4111 # int('0b1000000001111', 2) # 5 high
        ]

        # now we'll dynamically generate all the other
        # flushes (including straight flushes)
        flushes = []
        gen = self.get_lexographically_next_bit_sequence(int('0b11111', 2))

        # 1277 = number of high cards
        # 1277 + len(str_flushes) is number of hands with all cards unique rank
        for i in range(1277 + len(straight_flushes) - 1): # we also iterate over SFs
            # pull the next flush pattern from our generator
            f = next(gen)

            # if this flush matches perfectly any
            # straight flush, do not add it
            notSF = True
            for sf in straight_flushes:
                # if f XOR sf == 0, then bit pattern 
                # is same, and we should not add
                if not f ^ sf:
                    notSF = False

            if notSF:
                flushes.append(f)

        # we started from the lowest straight pattern, now we want to start ranking from
        # the most powerful hands, so we reverse
        flushes.reverse()

        # now add to the lookup map:
        # start with straight flushes and the rank of 1
        # since theyit is the best hand in poker
        # rank 1 = Royal Flush!
        rank = 1
        for sf in straight_flushes:
            prime_product = Card.prime_product_from_rankbits(sf)
            self.flush_lookup[prime_product] = rank
            rank += 1

        # we start the counting for flushes on max full house, which
        # is the worst rank that a full house can have (2,2,2,3,3)
        rank = LookupTable.MAX_FULL_HOUSE + 1
        for f in flushes:
            prime_product = Card.prime_product_from_rankbits(f)
            self.flush_lookup[prime_product] = rank
            rank += 1

        # we can reuse these bit sequences for straights
        # and high cards since they are inherently related
        # and differ only by context 
        self.straight_and_highcards(straight_flushes, flushes)

    def straight_and_highcards(self, straights, highcards):
        """
        Unique five card sets. Straights and highcards. 

        Reuses bit sequences from flush calculations.
        """
        rank = LookupTable.MAX_FLUSH + 1

        for s in straights:
            prime_product = Card.prime_product_from_rankbits(s)
            self.unsuited_lookup[prime_product] = rank
            rank += 1

        rank = LookupTable.MAX_PAIR + 1

        gen = self.get_lexographically_next_bit_sequence(int('0b111', 2))
        highcards.append(7)
        highcards.extend((next(gen) for _ in range(286 - 1)))
        highcards.sort(reverse=True)

        for h in highcards:
            prime_product = Card.prime_product_from_rankbits(h)
            self.unsuited_lookup[prime_product] = rank
            rank += 1

    def multiples(self):
        """
        Pair, Two Pair, Three of a Kind, Full House, and 4 of a Kind.
        """
        backwards_ranks = range(len(Card.INT_RANKS) - 1, -1, -1)

        # 1) Four of a Kind
        rank = LookupTable.MAX_STRAIGHT_FLUSH + 1

        # for each choice of a set of four rank
        for i in backwards_ranks:

            # and for each possible kicker rank
            kickers = list(backwards_ranks[:])
            kickers.remove(i)
            for k in kickers:
                product = Card.PRIMES[i]**4 * Card.PRIMES[k]
                self.unsuited_lookup[product] = rank
                rank += 1
        
        # 2) Full House
        rank = LookupTable.MAX_FOUR_OF_A_KIND + 1

        # for each three of a kind
        for i in backwards_ranks:

            # and for each choice of pair rank
            pairranks = list(backwards_ranks[:])
            pairranks.remove(i)
            for pr in pairranks:
                product = Card.PRIMES[i]**3 * Card.PRIMES[pr]**2
                self.unsuited_lookup[product] = rank
                rank += 1

        # 3) Three of a Kind
        rank = LookupTable.MAX_STRAIGHT + 1

        # pick three of one rank
        for r in backwards_ranks:

            kickers = list(backwards_ranks[:])
            kickers.remove(r)
            gen = itertools.combinations(kickers, 2)

            for kickers in gen:

                c1, c2 = kickers
                product = Card.PRIMES[r]**3 * Card.PRIMES[c1] * Card.PRIMES[c2]
                self.unsuited_lookup[product] = rank
                rank += 1

            self.unsuited_lookup[Card.PRIMES[r]**3] = rank
            rank += 1

        # 4) Two Pair
        rank = LookupTable.MAX_THREE_OF_A_KIND + 1

        tpgen = itertools.combinations(backwards_ranks, 2)
        for tp in tpgen:

            pair1, pair2 = tp
            kickers = list(backwards_ranks[:])
            kickers.remove(pair1)
            kickers.remove(pair2)
            for kicker in kickers:

                product = Card.PRIMES[pair1]**2 * Card.PRIMES[pair2]**2 * Card.PRIMES[kicker]
                self.unsuited_lookup[product] = rank
                rank += 1

        # 5) Pair
        rank = LookupTable.MAX_TWO_PAIR + 1

        # choose a pair
        for pairrank in backwards_ranks:

            kickers = list(backwards_ranks[:])
            kickers.remove(pairrank)

            for k1 in kickers:

                product = Card.PRIMES[pairrank]**2 * Card.PRIMES[k1]
                if len(left_kickers := tuple(k for k in kickers if k < k1)) < 2:

                    self.unsuited_lookup[product] = rank
                    rank += 1

                else:
                    for k2, k3 in itertools.combinations(left_kickers, 2):

                        self.unsuited_lookup[product * Card.PRIMES[k2] * Card.PRIMES[k3]] = rank
                        rank += 1

                    self.unsuited_lookup[product] = rank
                    rank += 1

    def write_table_to_disk(self, table, filepath):
        """
        Writes lookup table to disk
        """
        with open(filepath, 'w') as f:
            for prime_prod, rank in table.iteritems():
                f.write(str(prime_prod) +","+ str(rank) + '\n')

    def get_lexographically_next_bit_sequence(self, bits):
        """
        Bit hack from here:
        http://www-graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation

        Generator even does this in poker order rank 
        so no need to sort when done! Perfect.
        """
        t = (bits | (bits - 1)) + 1 
        next = t | ((int((t & -t) / (bits & -bits)) >> 1) - 1)
        yield next
        while True:
            t = (next | (next - 1)) + 1 
            next = t | ((int((t & -t) / (next & -next)) >> 1) - 1)
            yield next