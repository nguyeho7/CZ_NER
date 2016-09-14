#!/usr/bin/env python3
from NER_utils import *

use_case_test = ["<gt Asii>",
                 "<if u Tygra>",
                 "<P<pf Dubenka> <ps Králová>>",
                 "<P<pf J . > <ps Fučík>>",
                 "<P<pd Doc .> <pd Ing .> <pf Václav> <ps Hájek> , <pd CSc .>>",
                 "<if TOPIC , <s s . r . o .>>",
                 "<T<ty 1995> / <ty 1996>>",
                 "<s ZTP - P>",
                 "<ic Galerie u <ps Klicperů>> ( Divadlo ) : <P<pf Karel> <ps Sládek>>"]

res = [get_tag(token) for token in use_case_test]
print(use_case_test)
print(res)
