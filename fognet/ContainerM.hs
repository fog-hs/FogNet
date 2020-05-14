{-# Language
 TypeFamilies
,ConstraintKinds
,MultiParamTypeClasses
#-}

-- moadic varients of Container API, for mutable containers

module ContainerM (module ContainerM,module Container) where
import Container

-- unsafeRead :: PrimMonad m => MVector (PrimState m) a -> Int -> m a
class Indexed v => AccessibleM m v where
 accessM  :: v a ->  Index v  -> m a
 collectM :: v a -> [Index v] -> m [a]

-- unsafeWrite :: PrimMonad m => MVector (PrimState m) a -> Int -> a -> m ()
class Indexed v => InsertM m v where
 insertM  :: v a ->   Index v -> a   -> m ()
 replaceM :: v a -> [(Index v,   a)] -> m ()

-- unsafeModify :: PrimMonad m => MVector (PrimState m) a -> (a -> a) -> Int -> m ()
class Indexed v => ModifiableM m v where
 modifyM :: v a ->   Index v -> (a -> a)  -> m ()
 updateM :: v a -> [(Index v,  (a -> a))] -> m ()
