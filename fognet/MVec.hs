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

module MVec where -- (module MVec,module MVecPartOne) where

import Data.Proxy
import qualified Data.Vector.Mutable as M
import Control.Monad.ST

import Container
import BoundedInt
import Sized
import List


type RawMVec = M.MVector RealWorld 

----
-- Sized API for RawMVec

instance HasSize (RawMVec a) where
 type SizeType (RawMVec a) = Nat

instance Sized (RawMVec a) where
 getSize = toNat . M.length

----
-- Container API for RawMVec

instance Indexed RawMVec where
 type Index RawMVec = Int

-- unsafe updates used
-- can use safe updates and reimplement type checked bounds version
-- but at the moment, these are used by that implementation.

{-
unsafeRead :: PrimMonad m => MVector (PrimState m) a -> Int -> m a

unsafeWrite :: PrimMonad m => MVector (PrimState m) a -> Int -> a -> m ()

unsafeModify :: PrimMonad m => MVector (PrimState m) a -> (a -> a) -> Int -> m ()
-}

instance InsertM IO RawMVec where
 insertM  v i a = M.unsafeWrite v i a
 replaceM v xs = traverse (uncurry (insertM v)) xs >> return ()

instance ModifiableM IO RawMVec where
 modifyM v i f = M.unsafeModify v f i
 updateM v xs = traverse (uncurry (modifyM v)) xs >> return ()


instance AccessibleM IO RawMVec where
 accessM  v i  = M.unsafeRead v i
 collectM v ns = traverse (accessM v) ns

----
-- MVec

data MVec (n :: Nat) a = MVec (RawMVec a)

{-
place at end, use thaw

toIVec'' :: IsNat n => Proxy (n :: Nat) -> [a] -> MVec n a 
toIVec'' p xs | getSize xs == (getNat p) = MVec (V.fromList xs)
           | otherwise = error e
 where e = "\n\ttoIVec, passed wrong lengthed list; " ++ "\n\t of length; " ++ show (length xs) ++ "\n\t /= " ++ show (show (fromNat (getNat p)))

toIVec' :: IsList (List n a) => Proxy n -> [a] -> MVec n a 
toIVec' p = toIVec . (toList p)

toIVec :: IsList (List n a) => List n a -> MVec n a 
toIVec xs = MVec (V.fromList (toUnsized xs))
-}

----
-- Sized API for MVec

instance HasSize (MVec n a) where
 type SizeType (MVec n a) = Nat

instance IsNat n => Sized (MVec n a) where
 getSize _ = getNat (Proxy @n)

instance IsNat n => HasSized (RawMVec a) (n :: Nat) where
 type SizedVersion (RawMVec a) n = MVec n a

----
-- Container API for MVec

instance IsNat n => Indexed (MVec n) where
 type Index (MVec n) = BoundedInt n 

instance IsNat n => InsertM IO (MVec n) where
 insertM  (MVec v) (BoundedInt i) a = insertM v i a 
 replaceM (MVec v) xs = replaceM v xs'
  where
   xs' = let (ys,zs) = unzip xs in zip (fromBoundedInts (Proxy @n) ys) zs

instance IsNat n => ModifiableM IO (MVec n) where
 modifyM (MVec v) (BoundedInt i) f = accessM v i >>= (insertM v i) . f
 updateM (MVec v) xs = updateM v xs'
  where
   xs' = let (ys,zs) = unzip xs in zip (fromBoundedInts (Proxy @n) ys) zs
-- ALERT test fusion of zip and unzip

instance IsNat n => AccessibleM IO (MVec n) where
 accessM  (MVec v) (BoundedInt i) = accessM v i
 collectM (MVec v)  xs = collectM v (fromBoundedInts (Proxy @n) xs)



 

instance IsNat n => ToSized (RawMVec a) n where
 toSized n | getNat (Proxy @n) == n = MVec 
           | otherwise = error "toSized given wrong Nat as size, use MVec constructor directly"

instance IsNat n => HasUnsized (MVec n a) where
 type UnsizedVersion (MVec n a) = RawMVec a

instance IsNat n => ToUnsized (MVec n a) where
 toUnsized (MVec v) = v

type family IsMVec x where
 IsMVec (MVec n a) = (IsNat n,ToUnsized (MVec n a),ToSized (RawMVec a) n)



