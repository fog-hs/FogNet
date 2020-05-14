{-# Language
TypeFamilies
#-}

module RawColumns where

import Sized
import Nat

newtype RawColumns a = RawColumns [[a]]

instance HasSize (RawColumns a) where
 type SizeType (RawColumns a) = [Nat]

instance Sized (RawColumns a) where
 getSize (RawColumns a) = map (toNat . length) a



