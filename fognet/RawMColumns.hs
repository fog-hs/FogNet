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

module RawMColumns where

import Data.Proxy

import MVec
import Container
import BoundedInt
import Sized
import List
import Columns

newtype RawMColumns a = RawMColumns (RawMVec (RawMVec a))
{-
toRawIColumns :: [[a]] -> RawMColumns a
toRawIColumns = RawMColumns . toRawMVec . map toRawMVec 
-}
----
-- Sized API for RawMColumns 

instance HasSize (RawMColumns cs) where
 type SizeType (RawMColumns cs) = [Nat]

instance Sized (RawMColumns cs) where
 getSize (RawMColumns cs) = error "cant getSize of Mthings, need monad"
-- map getSize $ fromRawMVec cs


----
-- Container API for RawMColumns 
{-
instance Indexed RawMColumns where
 type Index RawMColumns = (Int,Int)

instance Insert RawMColumns where
 insert  (RawMColumns v) (i,j) a = RawMColumns $modify v i (\w -> insert w j a)
 replace v xs = foldr (\(ij,a) v' -> insert v' ij a) v xs

instance Modifiable RawMColumns where
 modify (RawMColumns v) (i,j) f = RawMColumns $ modify v i (\w -> modify w j f)
 update v xs = foldr (\(ij,f) v' -> modify v' ij f) v xs

instance Accessible RawMColumns where
 access (RawMColumns v) (i,j) = access (access v i) j
 collect v xs = map (\ij -> access v ij) xs
-}