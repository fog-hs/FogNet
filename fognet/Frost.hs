{-# Language
 RankNTypes
,KindSignatures
,TypeApplications
,ScopedTypeVariables
,DataKinds
#-}

--module Frost where

import Nat
import BoundedInt

import Data.Proxy
import qualified Data.Vector as V
import qualified Data.Vector.Mutable as M 

import Container
import MVec
import IVec

thaw :: forall n a. IsNat n => IVec n a -> IO (MVec n a)
thaw (IVec v) = fmap MVec $ V.thaw v

freeze :: forall n a. IsNat n => MVec n a -> IO (IVec n a)
freeze (MVec v) = fmap IVec $ V.freeze v

unfreezeLoop :: forall i (s :: *) a b c. IsNat i 
 => (MVec i a -> IO b)
 -> (IVec i a -> IO Bool) 
 -> IVec i a 
 -> IO b
unfreezeLoop f g v' = thaw v' >>= h
 where
  h v = do
   f v
   b <- freeze v >>= g
   if b then h v else return undefined

-- starts by thawing the input vector
-- then opperates on it in the mutable setting
-- and freezes it to do opperations that do not require mutation,
-- which can be used eg. for writing a saved state to disk
-- todo; can fork a thread to allow mutation to happen in parallel
testUnfreezeLoopIO :: IO ()
testUnfreezeLoopIO = unfreezeLoop 
                      (\mv -> (modifyM mv (BoundedInt 1) succ) >> return undefined) 
                      (\v -> print (toList v) >> (if (sumV v) > 30 then return False else return True))
                      (toIVec' (Proxy @(ToNat 5)) [1::Int ..5])
 where
  toList :: IVec n Int -> [Int]
  toList (IVec v) = V.toList v
  sumV :: forall n. IVec n Int -> Int
  sumV = sum . toList 

main = testUnfreezeLoopIO 