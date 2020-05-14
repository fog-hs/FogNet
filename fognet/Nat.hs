{-# Language
 KindSignatures
,TypeOperators
,DataKinds
,PolyKinds
,GADTs
,TypeFamilies
,UndecidableInstances
,TypeFamilyDependencies
,FlexibleInstances
,RankNTypes,TypeApplications
,ScopedTypeVariables
,ConstraintKinds
,UndecidableSuperClasses
#-}

module Nat where

import Data.Proxy
import GHC.Exts 
import qualified GHC.TypeNats

import TyFun

data family Sing (a :: k)

class SingI (a :: k) where
 sing :: Sing a

class SingE (a :: k) where
 type Demote a :: *
 fromSing :: Sing a -> Demote (Any :: k)

class (SingI a,SingE a) => SingRep a
instance (SingI a,SingE a) => SingRep a

--

type Nat = [()]

type IsNat (n :: Nat) = SingI n

data IsNatSym :: Nat ~> Constraint
type instance Apply IsNatSym n = IsNat n

type AreNats (ns :: [Nat]) = AllConstraints IsNatSym ns

showNat :: Nat -> String
showNat = show . length

toNat :: Int -> Nat
toNat n = replicate n ()

fromNat :: Nat -> Int
fromNat = length

type Zero = ('[] :: Nat)
zero' :: Nat
zero' = []

type Succ = (:) '()
succ' :: Nat -> Nat
succ' = (:) ()

instance Enum Nat where
 succ = succ'
 pred [] = error "pred Zero"
 pred xs = tail xs
 toEnum = toNat
 fromEnum = fromNat

data instance Sing (a :: Nat) where
 SingZero :: Sing Zero
 SingSucc :: SingRep n => Sing n -> Sing (Succ n)

instance SingI Zero where
 sing = SingZero

instance SingRep n => SingI (Succ n) where
 sing = SingSucc sing

instance SingE (a :: Nat) where
 type Demote a = Nat
 fromSing SingZero = []
 fromSing (SingSucc n) = () : (fromSing n)

class AreNats ns => GetNats ns where
 getNats :: Proxy ns -> [Nat]

instance GetNats '[] where
 getNats _ = []

instance (AreNats (n ': ns),GetNats ns) => GetNats (n ': ns) where
 getNats _ = fromSing (sing :: Sing n) : getNats (Proxy @ns)

getNat :: forall n. IsNat n => Proxy n -> Nat
getNat _ = fromSing (sing :: Sing n)

-- dont write putNat, cant get type from value, use Proxy
{-
*Main> fromSing (sing :: Sing Zero)
[]
*Main> fromSing (sing :: Sing (Succ Zero))
[()]
-}

type family ToNat (n :: GHC.TypeNats.Nat) :: Nat where
 ToNat 0 = Zero
 ToNat n = Succ (ToNat (n GHC.TypeNats.- 1))
 

