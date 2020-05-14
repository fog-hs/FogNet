{-# Language
 TypeApplications
,RankNTypes
,DataKinds
,KindSignatures
,ScopedTypeVariables
,TypeFamilies


#-}

module BoundedInt (module BoundedInt,module Nat) where

import Nat
import Data.Proxy

class Bound x where
 type Bounds x
 getBounds :: x -> Bounds x

newtype BoundedInt (n :: Nat) = BoundedInt Int deriving Show

instance SingI n => Bound (BoundedInt n) where
 type Bounds (BoundedInt n) = Nat
 getBounds (_ :: BoundedInt n) = fromSing (sing :: Sing n)

toBoundedInt' :: Sing n -> Int -> BoundedInt (n :: Nat)
toBoundedInt' n i | (fromNat . fromSing) n > i = BoundedInt i :: BoundedInt n
                  | otherwise = error $ "Int `i` out of bounds in; toBoundedInt " ++ show ((fromNat . fromSing) n) ++ " " ++ show i

toBoundedInt :: forall n. SingI n => Int -> BoundedInt (n :: Nat)
toBoundedInt = toBoundedInt' sing

fromBoundedInt :: BoundedInt n -> Int
fromBoundedInt (BoundedInt i) = i

egBoundedInt :: BoundedInt (ToNat 5)
egBoundedInt = toBoundedInt 3 

testBoundedInt = let i = toBoundedInt @(ToNat 5) 3 in (fromBoundedInt i,fromNat $ getBounds i)

fromBoundedInts :: IsNat n => Proxy n -> [BoundedInt n] -> [Int]
fromBoundedInts _ [] = []
fromBoundedInts p ((BoundedInt i :: BoundedInt n):xs) = i : fromBoundedInts p xs
--fromBoundedInts _ _ = error "fromBoundedInts, not all the same bound"

