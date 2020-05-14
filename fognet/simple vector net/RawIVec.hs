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

module RawIVec where

import qualified Data.Vector as V
import Container
import Sized
import Nat

type RawIVec = V.Vector 

fromRawIVec :: RawIVec a -> [a]
fromRawIVec = V.toList

toRawIVec :: [a] -> RawIVec a
toRawIVec = V.fromList

----
-- Sized API for RawIVec

instance HasSize (RawIVec a) where
 type SizeType (RawIVec a) = Nat

instance Sized (RawIVec a) where
 getSize = toNat . V.length

----
-- Container API for RawIVec

instance Indexed RawIVec where
 type Index RawIVec = Int

-- unsafe updates used
-- can use safe updates and reimplement type checked bounds version
-- but at the moment, these are used by that implementation.
instance Insert RawIVec where
 insert  v i a = replace v [(i,a)]
 replace v xs = V.unsafeUpd v xs

instance Modifiable RawIVec where
 modify v i f = insert v i (f (access v i))
 update v [] = v
 update v ((i,f):xs) = update (modify v i f) xs

instance Accessible RawIVec where
 access  v i      = V.unsafeIndex v i
 collect v []     = []
 collect v (n:ns) = access v n : collect v ns

instance CPrelude RawIVec where
 mapC = V.map 
 imapC = V.imap
 zipWithC = V.zipWith
 foldrC = V.foldr
 ifoldrC = V.ifoldr
 sumC = V.sum
