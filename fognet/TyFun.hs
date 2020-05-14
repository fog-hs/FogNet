{-# Language
 KindSignatures
,TypeOperators
,DataKinds
,PolyKinds
,GADTs
,TypeFamilies
,UndecidableInstances
,TypeFamilyDependencies
,AllowAmbiguousTypes
,FlexibleInstances
,RankNTypes,TypeApplications
,ScopedTypeVariables
,ConstraintKinds
#-}

module TyFun where

import GHC.Exts

data TyFun :: * -> * -> *

type a ~> b = TyFun a b -> *
infixr 0 ~>

type family Apply (f :: k1 ~> k2) (x :: k1) = (k :: k2) 

type a @@ b = Apply a b
infixl 9 @@

data TyCon2 (tc :: a -> b -> c) (tf :: TyFun a (b ~> c))
data TyCon1 (tc :: a -> b) (tf :: TyFun a b)

type instance Apply (TyCon1 tc) a = tc a
type instance Apply (TyCon2 tc) a = TyCon1 (tc a)

--

data TyFun' :: * -> * -> *

type a ~>> b = TyFun' a b -> *
infixr 0 ~>>

type family Apply' (f :: k1 ~>> k2) (x :: k1) = (k :: k2) | k -> x f

type a @@@ b = Apply' a b
infixl 9 @@@

--data TyCon2' (tc :: a -> b -> c) (tf :: TyFun' a (b ~>> c))
data TyCon1' (tc :: a -> b) (tf :: TyFun' a b)

type instance Apply' (TyCon1' tc) a = tc a
--type instance Apply' (TyCon2' tc) a = TyCon1' (tc a)

{-
data ApplySym0 :: (k1 ~> k2) ~> k1 ~> k2
type instance Apply ApplySym0 a = ApplySym1 a

data ApplySym1 :: (k1 ~> k2) -> k1 ~> k2 
type instance Apply (ApplySym1 a) b = Apply a b

{-
data TyCon2Sym0 :: (tc :: a -> b -> c) ~> (tf :: TyFun a (b ~> c)) ~> *
type instance Apply (TyCon2Sym0 tc) a = TyCon2 
-}
{-
data TyCon1Sym0 :: (tc :: a -> b) ~> (tf :: TyFun a b) ~> *
type instance Apply TyCon1Sym0 a = TyCon1Sym1 a

data TyCon1Sym1 :: (tc :: a -> b) -> (tf :: TyFun a b) ~> *
type instance Apply (TyCon1Sym1 a) b = TyCon1 a b
-}
-}

-- Compose



data (:.$) :: (b ~> c) ~> (a ~> b) ~> (a ~> c)
type instance Apply (:.$) a = (:.$$) a

data (:.$$) :: (b ~> c) -> (a ~> b) ~> (a ~> c)
type instance Apply ((:.$$) a) b = (:.$$$) a b

data (:.$$$) :: (b ~> c) -> (a ~> b) -> (a ~> c)
type instance Apply ((:.$$$) a b) x = (:.$$$$) a b x

type (:.$$$$) a b x = (:.) a b x

type family (:.) (f :: b ~> c) (g :: a ~> b) (x :: a) :: c where 
 (:.) f g x = f @@ (g @@ x)

-- mapping constraints

type family AllConstraints (f :: k ~> Constraint) (xs :: [k]) :: Constraint where
 AllConstraints _ '[] = ()
 AllConstraints f (x ': xs) = (f @@ x,AllConstraints f xs)
