{-# Language 
  TypeFamilies
 ,FlexibleContexts
 ,MultiParamTypeClasses
 ,ScopedTypeVariables
 ,RankNTypes
 ,AllowAmbiguousTypes
 ,PolyKinds
 ,DataKinds
 #-}

module Sized where

class HasSize (v :: *) where
 type SizeType v

class (HasSize v) => Sized (v :: *) where
 getSize :: v -> SizeType v

class (SizeType v ~ SizeType (SizedVersion v n),Sized v,Sized (SizedVersion v n)) => HasSized (v :: *) (n :: SizeType v) where
 type SizedVersion v n :: *

class HasSized v n => ToSized (v :: *) n where
 toSized :: SizeType (SizedVersion v n) -> v -> SizedVersion v n

class Sized v => HasUnsized (v :: *) where
 type UnsizedVersion v :: *

class HasUnsized v => ToUnsized (v :: *) where
 toUnsized :: v -> UnsizedVersion v