{-# Language
 RankNTypes
,UndecidableInstances
,TypeApplications
,KindSignatures
,ScopedTypeVariables
,DataKinds
,GADTs
,FlexibleInstances
,FlexibleContexts
,TypeOperators
,PolyKinds
,TypeFamilies
,MultiParamTypeClasses
#-}

-- NB.
-- cant infer size from unsized,
-- but can throuw runtime error if user specified size is wrong.

module List where

import Data.Proxy

import Sized
import Nat

type family Length (xs :: [a]) :: Nat where
 Length '[] = Zero
 Length (x ': xs) = Succ (Length xs)

type family (!!) (xs :: [a]) (n :: Nat) :: a where
 (x ': xs) !! Zero = x
 (x ': xs) !! (Succ n) = xs !! n

 -- '[] !! _ = error

instance HasSize [a] where
 type SizeType [a] = Nat
 
instance Sized [a] where
 getSize = toNat . length

data List (n :: Nat) a where
 Empty :: List Zero a
 Cons :: a -> List n a -> List (Succ n) a

instance Functor (List Zero) where
 fmap f Empty = Empty 

instance Functor (List n) => Functor (List (Succ n)) where
 fmap f (Cons x xs) = Cons (f x) (fmap f xs)



instance (Show a,IsList (List n a)) => Show (List n a) where
 show xs = show $ fromList xs

toList :: forall n a. IsList (List n a)=> Proxy (n :: Nat) -> [a] -> List n a
toList _ = toSized (getNat (Proxy @n))

fromList :: IsList (List n a) => List n a -> [a]
fromList = toUnsized


instance HasSize (List n a) where
 type SizeType (List n a) = Nat

instance IsNat n => Sized (List n a) where
 getSize _ = getNat (Proxy @n)

instance IsNat n => HasSized [a] (n :: Nat) where
 type SizedVersion [a] n = List n a

instance ToSized [a] Zero where
 toSized [] [] = Empty
 toSized _ []  = error "wrong size provided in toSized"
 toSized _ _ = error "wrong size provided in toSized!"


instance (IsNat n,ToSized [a] n) => ToSized [a] (Succ n) where
 toSized (() : ns)(x : xs) = x `Cons` (toSized ns xs)
 toSized _ []  = error "wrong size provided in toSized!!"
 toSized _ _ = error "wrong size provided in toSized!!!"



instance IsNat n => HasUnsized (List n a) where
 type UnsizedVersion (List n a) = [a]

instance ToUnsized (List Zero a) where
 toUnsized Empty = []

instance (IsNat n,ToUnsized (List n a)) => ToUnsized (List (Succ n) a) where
 toUnsized (Cons x xs) = x : (toUnsized xs)

type family IsList x where
 IsList (List n a) = (Functor (List n),IsNat n,ToUnsized (List n a),ToSized [a] n)

{-
*Main> toUnsized $ (toSized [] [] :: List Zero a)
[]
-}

{-
*Main> toUnsized $ (toSized [()] [1] :: List (ToNat 1) Int)
[1]
-}

{-
*Main> toUnsized $ (toSized [()] [] :: List (ToNat 1) Int)
*** Exception: wrong size provided in toSized!!
CallStack (from HasCallStack):
  error, called at C:\\Users\Desktop\Desktop\dj\neural net v2\restart\List.hs:35:18 in main:Main

*Main> toUnsized $ (toSized [()] [1] :: List (ToNat 0) Int)
*** Exception: wrong size provided in toSized!
CallStack (from HasCallStack):
  error, called at C:\\Users\Desktop\Desktop\dj\neural net v2\restart\List.hs:30:16 in main:Main

*Main> toUnsized $ (toSized [] [1] :: List (ToNat 1) Int)
*** Exception: wrong size provided in toSized!!!
CallStack (from HasCallStack):
  error, called at C:\\Users\Desktop\Desktop\dj\neural net v2\restart\List.hs:36:16 in main:Main
-}