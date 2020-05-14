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

type Container c = (Accessible c,Insert c,Modifiable c)
