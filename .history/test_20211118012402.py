"""
An original string, consisting of lowercase English letters, can be encoded by the following steps:

Arbitrarily split it into a sequence of some number of non-empty substrings.
Arbitrarily choose some elements (possibly none) of the sequence, and replace each with its length (as a numeric string).
Concatenate the sequence as the encoded string.
For example, one way to encode an original string "abcdefghijklmnop" might be:

Split it as a sequence: ["ab", "cdefghijklmn", "o", "p"].
Choose the second and third elements to be replaced by their lengths, respectively. The sequence becomes ["ab", "12", "1", "p"].
Concatenate the elements of the sequence to get the encoded string: "ab121p".
Given two encoded strings s1 and s2, consisting of lowercase English letters and digits 1-9 (inclusive), return true if there exists an original string that could be encoded as both s1 and s2. Otherwise, return false.

"""
class Solution:
    def originalDigits(self, s: str) -> str:
        s = s.replace('z', '2')
        s = s.replace('w', '2')
        s = s.replace('u', '4')
        s = s.replace('x', '6')
        s = s.replace('g', '8')
        s = s.replace('o', '1')
        s = s.replace('s', '5')
        s = s.replace('f', '4')
        s = s.replace('h', '3')
        s = s.replace('i', '1')
        s = s.replace('e', '3')
        s = s.replace('t', '7')
        s = s.replace('l', '1')
        s = s.replace('n', '5')
        s = s.replace('r', '1')
        s = s.replace('d', '3')
        s = s.replace('v', '7')
        s = s.replace('m', '2')
        s = s.replace('c', '2')
        s = s.replace('p', '7')
        s = s.replace('b', '1')
        s = s.replace('y', '7')
        s = s.replace('a', '1')
        s = s.replace('k', '5')
        s = s.replace('q', '9')
        s = s.replace('j', '8')
        s = s.replace('z', '2')
        return s
    
        