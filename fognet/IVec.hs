{-# Language 
 ScopedTypeVariables
,RankNTypes
,TypeApplications
,KindSignatures
,TypeSynonymInstances
,DataKinds
,TypeFamilies
,FlexibleInstances
,MultiParamTypeClasses
#-}

module IVec where

import Data.Proxy
import qualified Data.Vector as V
import Control.Monad.ST
import RawIVec

import Container
import BoundedInt
import Sized
import List

----
-- IVec

data IVec (n :: Nat) a = IVec (RawIVec a)

toIVec'' :: IsNat n => Proxy (n :: Nat) -> [a] -> IVec n a 
toIVec'' p xs | getSize xs == (getNat p) = IVec (V.fromList xs)
           | otherwise = error e
 where e = "\n\ttoIVec, passed wrong lengthed list; " ++ "\n\t of length; " ++ show (length xs) ++ "\n\t /= " ++ show (show (fromNat (getNat p)))

toIVec' :: IsList (List n a) => Proxy n -> [a] -> IVec n a 
toIVec' p = toIVec . (toList p)

toIVec :: IsList (List n a) => List n a -> IVec n a 
toIVec xs = IVec (V.fromList (toUnsized xs))

----
-- Sized API for IVec

instance HasSize (IVec n a) where
 type SizeType (IVec n a) = Nat

instance IsNat n => Sized (IVec n a) where
 getSize _ = getNat (Proxy @n)

instance IsNat n => HasSized (RawIVec a) (n :: Nat) where
 type SizedVersion (RawIVec a) n = IVec n a

----
-- Container API for IVec

instance IsNat n => Indexed (IVec n) where
 type Index (IVec n) = BoundedInt n 

instance IsNat n => Insert (IVec n) where
 insert  (IVec v) (BoundedInt i) a = IVec $ insert v i a 
 replace (IVec v) xs = IVec $ replace v xs'
  where
   xs' = let (ys,zs) = unzip xs in zip (fromBoundedInts (Proxy @n) ys) zs


instance IsNat n => Modifiable (IVec n) where
 modify (IVec v) (BoundedInt i) f = IVec $ insert v i (f (access v i))
 update (IVec v) xs = IVec $ update v xs'
  where
   xs' = let (ys,zs) = unzip xs in zip (fromBoundedInts (Proxy @n) ys) zs
-- ALERT test fusion of zip and unzip

instance IsNat n => Accessible (IVec n) where
 access  (IVec v) (BoundedInt i) = access v i
 collect (IVec v)  xs = collect v (fromBoundedInts (Proxy @n) xs)

instance IsNat n => ToSized (RawIVec a) n where
 toSized n | getNat (Proxy @n) == n = IVec 
           | otherwise = error "toSized given wrong Nat as size, use IVec constructor directly"

instance IsNat n => HasUnsized (IVec n a) where
 type UnsizedVersion (IVec n a) = RawIVec a

instance IsNat n => ToUnsized (IVec n a) where
 toUnsized (IVec v) = v

type family IsIVec x where
 IsIVec (IVec n a) = (IsNat n,ToUnsized (IVec n a),ToSized (RawIVec a) n)

