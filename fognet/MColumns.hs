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

module MColumns where

import Data.Proxy

import MVec
import Container
import BoundedInt
import Sized
import List
import Columns
import RawMColumns

----
-- MColumns

data MColumns (n :: [Nat]) a = MColumns (RawMVec (RawMVec a))
{-
toMColumns :: GetNats ns => Proxy (ns :: [Nat]) -> RawMColumns a ->MColumns ns a 
toMColumns _ (RawMColumns x) = MColumns x

toMColumns' :: GetNats ns =>  Proxy (ns :: [Nat]) -> [[a]] -> MColumns ns a 
toMColumns' p = toMColumns p . toRawMColumns 
-}

----
-- Sized API for MColumns

instance HasSize (MColumns lengths a) where
 type SizeType (MColumns lengths a) = [Nat]

instance GetNats lengths => Sized (MColumns lengths a) where
 getSize _ = getNats (Proxy @lengths)

instance GetNats lengths => HasSized (RawMColumns a) (lengths :: [Nat]) where
 type SizedVersion (RawMColumns a) lengths = MColumns lengths a

{-
instance GetNats lengths => ToSized (RawMColumns a) lengths where
 toSized ns | getNats (Proxy @lengths) == ns = toMColumns (Proxy @lengths) -- MColumns 
           | otherwise = error "toSized given wrong Nats as lengths, use MColumns constructor directly"
-}
instance GetNats lengths => HasUnsized (MColumns lengths a) where
 type UnsizedVersion (MColumns lengths a) = RawMColumns a

instance GetNats lengths => ToUnsized (MColumns lengths a) where
 toUnsized (MColumns v) = RawMColumns v

type family IsMColumns x where
 IsMColumns (MColumns lengths a) = (GetNats lengths,ToUnsized (MColumns lengths a),ToSized (RawMColumns a) lengths)


----
-- Container API for MVec
{-
data DoubleBoundedInt (n1 :: Nat) (n2 :: Nat) = DoubleBoundedInt (Int,Int)

toDoubleBoundedInt' :: (IsNat n1,IsNat n2) => BoundedInt n1 -> BoundedInt n2 -> DoubleBoundedInt n1 n2
toDoubleBoundedInt' (BoundedInt i1) (BoundedInt i2) = DoubleBoundedInt (i1,i2)

toDoubleBoundedInt :: (IsNat n1,IsNat n2) => (Int,Int) -> DoubleBoundedInt n1 n2
toDoubleBoundedInt (i1,i2) = toDoubleBoundedInt' (toBoundedInt i1) (toBoundedInt i2)

type family ToDoubleBoundedInt (lengths :: [Nat]) where
-}

data DoubleBoundedInt (ns :: [Nat]) = DoubleBoundedInt (Int,Int)

toDoubleBoundedInt :: forall ns. (IsNat (Length ns),GetNats ns) => Proxy ns -> (Int,Int) ->DoubleBoundedInt ns
toDoubleBoundedInt p ij@(i,j) | check = DoubleBoundedInt ij
   | otherwise = error e2
 where
  e2 = "toDoubleBoundedInt error, second index out of bounds; " ++ (show $ map fromNat (getNats p)) ++ " @ " ++ (show ij)
  e1 = "toDoubleBoundedInt error, first index out of bounds; " ++ (show bound1) ++ " <= " ++ (show i)
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
instance GetNats lengths => Indexed (MColumns lengths) where
 type Index (MColumns lengths) = DoubleBoundedInt lengths

--holy shit!!! 
--need to use the first int to lookup the length of the inner list.
-- needs singleton bounded int...
-- and that should be a nat anyway
-- aaand
-- how to use the first in the secod of the pair
-- oh just ue a type family... 


{-
instance IsNat n => Insert (MColumns n) where
 insert  (MColumns v) (BoundedInt i) a = MVec $ insert v i a 
 replace (MColumns v) xs = MVec $ replace v xs'
  where
   xs' = let (ys,zs) = unzip xs in zip (fromBoundedInts (Proxy @n) ys) zs


instance IsNat n => Modifiable (MColumns n) where
 modify (MColumns v) (BoundedInt i) f = MVec $ insert v i (f (access v i))
 update (MColumns v) xs = MVec $ update v xs'
  where
   xs' = let (ys,zs) = unzip xs in zip (fromBoundedInts (Proxy @n) ys) zs
-- ALERT test fusion of zip and unzip

instance IsNat n => Accessible (MColumns n) where
 access  (MColumns v) (BoundedInt i) = access v i
 collect (MColumns v)  xs = collect v (fromBoundedInts (Proxy @n) xs)
-}