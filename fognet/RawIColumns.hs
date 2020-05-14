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

module RawIColumns where

import Data.Proxy

import IVec
import Container
import BoundedInt
import Sized
import List
import Columns

newtype RawIColumns a = RawIColumns (RawIVec (RawIVec a))

toRawIColumns :: [[a]] -> RawIColumns a
toRawIColumns = RawIColumns . toRawIVec . map toRawIVec 

----
-- Sized API for RawIColumns 

instance HasSize (RawIColumns cs) where
 type SizeType (RawIColumns cs) = [Nat]

instance Sized (RawIColumns cs) where
 getSize (RawIColumns cs) = map getSize $ fromRawIVec cs


----
-- Container API for RawIColumns 

instance Indexed RawIColumns where
 type Index RawIColumns = (Int,Int)

instance Insert RawIColumns where
 insert  (RawIColumns v) (i,j) a = RawIColumns $modify v i (\w -> insert w j a)
 replace v xs = foldr (\(ij,a) v' -> insert v' ij a) v xs

instance Modifiable RawIColumns where
 modify (RawIColumns v) (i,j) f = RawIColumns $ modify v i (\w -> modify w j f)
 update v xs = foldr (\(ij,f) v' -> modify v' ij f) v xs

instance Accessible RawIColumns where
 access (RawIColumns v) (i,j) = access (access v i) j
 collect v xs = map (\ij -> access v ij) xs
