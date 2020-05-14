{-# Language
 TypeFamilies
,ConstraintKinds
,MultiParamTypeClasses
#-}
module Container where

class Indexed (v :: * -> *) where
 type Index v :: *
 
class Indexed v => Accessible v where
 access  :: v a ->  Index v  -> a
 collect :: v a -> [Index v] -> [a]

class Indexed v => Insert v where
 insert  :: v a -> Index v -> a  -> v a
 replace :: v a -> [(Index v, a)]  -> v a

class Indexed v => Modifiable v where
 modify :: v a -> Index v -> (a -> a)  -> v a
 update :: v a -> [(Index v,(a -> a))] -> v a

class Indexed v => CPrelude v where
 mapC :: (a -> b) -> v a -> v b
 imapC :: (Index v -> a -> b) -> v a -> v b
 zipWithC :: (a -> b -> c) -> v a -> v b -> v c 
 foldrC :: (a -> b -> b) -> b -> v a -> b
 ifoldrC :: (Int -> a -> b -> b) -> b -> v a -> b
 sumC :: Num a => v a -> a
 

type Container c = (CPrelude c,Accessible c,Insert c,Modifiable c)
