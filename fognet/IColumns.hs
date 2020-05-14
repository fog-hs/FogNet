{-# Language 
 ScopedTypeVariables
,FlexibleContexts
,RankNTypes
,TypeApplications
,KindSignatures
,TypeSynonymInstances
,DataKinds
,TypeFamilies
,FlexibleInstances
,MultiParamTypeClasses
#-}

module IColumns where

import Data.Proxy

import IVec
import Container
import BoundedInt
import Sized
import List
import Columns
import RawIColumns

----
-- IColumns

data IColumns (n :: [Nat]) a = IColumns (RawIVec (RawIVec a))

toIColumns :: GetNats ns => Proxy (ns :: [Nat]) -> RawIColumns a ->IColumns ns a 
toIColumns _ (RawIColumns x) = IColumns x

toIColumns' :: GetNats ns =>  Proxy (ns :: [Nat]) -> [[a]] -> IColumns ns a 
toIColumns' p = toIColumns p . toRawIColumns 


----
-- Sized API for IColumns

instance HasSize (IColumns lengths a) where
 type SizeType (IColumns lengths a) = [Nat]

instance GetNats lengths => Sized (IColumns lengths a) where
 getSize _ = getNats (Proxy @lengths)

instance GetNats lengths => HasSized (RawIColumns a) (lengths :: [Nat]) where
 type SizedVersion (RawIColumns a) lengths = IColumns lengths a


instance GetNats lengths => ToSized (RawIColumns a) lengths where
 toSized ns | getNats (Proxy @lengths) == ns = toIColumns (Proxy @lengths) -- IColumns 
           | otherwise = error "toSized given wrong Nats as lengths, use IColumns constructor directly"

instance GetNats lengths => HasUnsized (IColumns lengths a) where
 type UnsizedVersion (IColumns lengths a) = RawIColumns a

instance GetNats lengths => ToUnsized (IColumns lengths a) where
 toUnsized (IColumns v) = RawIColumns v

type family IsIColumns x where
 IsIColumns (IColumns lengths a) = (GetNats lengths,ToUnsized (IColumns lengths a),ToSized (RawIColumns a) lengths)


----
-- Container API for IVec
{-
data DoubleBoundedInts (n1 :: Nat) (n2 :: Nat) = DoubleBoundedInts (Int,Int)

toDoubleBoundedInts' :: (IsNat n1,IsNat n2) => BoundedInt n1 -> BoundedInt n2 -> DoubleBoundedInts n1 n2
toDoubleBoundedInts' (BoundedInt i1) (BoundedInt i2) = DoubleBoundedInts (i1,i2)

toDoubleBoundedInts :: (IsNat n1,IsNat n2) => (Int,Int) -> DoubleBoundedInts n1 n2
toDoubleBoundedInts (i1,i2) = toDoubleBoundedInts' (toBoundedInt i1) (toBoundedInt i2)

type family ToDoubleBoundedInts (lengths :: [Nat]) where
-}

data DoubleBoundedInts (ns :: [Nat]) = DoubleBoundedInts (Int,Int)

toDoubleBoundedInts :: forall ns. (IsNat (Length ns),GetNats ns) => Proxy ns -> (Int,Int) ->DoubleBoundedInts ns
toDoubleBoundedInts p ij@(i,j) | check = DoubleBoundedInts ij
   | otherwise = error e2
 where
  e2 = "toDoubleBoundedInts error, second index out of bounds; " ++ (show $ map fromNat (getNats p)) ++ " @ " ++ (show ij)
  e1 = "toDoubleBoundedInts error, first index out of bounds; " ++ (show bound1) ++ " <= " ++ (show i)
  check = if fromNat bound1 <= i then error e1 else j <= fromNat bound2
  bound1 = getNat (Proxy @(Length ns)) 
  bound2 = (getNats p) !! i
-- a pair of BoundedInts, where the second is Bound
-- by lengths !! fst
-- takes the [Nat] and the type other BoundedInts
-- with the first being bound by the length of the lengths
-- actually, take a pair of Nats
-- no thats slow
-- can cast from Int to Nat at type level instead of Nat to Int as value
-- so instead of the first bounded int, a type level n and a nat
-- and a second int, but with all this as a datatype
-- that performs the bound checking on construction
-- ahh
-- right, instead of a pair of bounded ints
-- its just a pair of ints with the list of nats as a phantom
-- as this can be used to infer both the length and second bound
-- on construction!
instance GetNats lengths => Indexed (IColumns lengths) where
 type Index (IColumns lengths) = DoubleBoundedInts lengths

--holy shit!!! 
--need to use the first int to lookup the length of the inner list.
-- needs singleton bounded int...
-- and that should be a nat anyway
-- aaand
-- how to use the first in the secod of the pair
-- oh just ue a type family... 

instance GetNats lengths => Insert (IColumns lengths) where
 insert  (IColumns v) (DoubleBoundedInts (i,j)) a = IColumns $modify v i (\w -> insert w j a)
 replace v xs = foldr (\(ij,a) v' -> insert v' ij a) v xs

instance GetNats lengths => Modifiable (IColumns lengths) where
 modify (IColumns v) (DoubleBoundedInts (i,j)) f = IColumns $ modify v i (\w -> modify w j f)
 update v xs = foldr (\(ij,f) v' -> modify v' ij f) v xs

instance GetNats lengths => Accessible (IColumns lengths) where
 access (IColumns v) (DoubleBoundedInts (i,j)) = access (access v i) j
 collect v xs = map (\ij -> access v ij) xs
