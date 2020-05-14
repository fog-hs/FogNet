{-# Language 
 PolyKinds 
,ConstraintKinds
,UndecidableInstances
,FlexibleInstances
,UndecidableSuperClasses
,TypeApplications
,AllowAmbiguousTypes
,ScopedTypeVariables
,TypeOperators
,TypeSynonymInstances
,MultiParamTypeClasses
,RankNTypes
,GADTs
,FlexibleContexts
,TypeFamilies
,DataKinds 
,StandaloneKindSignatures
#-}

import Data.Proxy
import GHC.Exts (Constraint)

import Container
import Sized
import List
import BoundedInt
import Columns
import IVec
import IColumns

type RawPList (n :: k) (a :: k -> *) = [a n]

type PList (m :: Nat) (n :: k) (a :: k -> *) = List m (a n)
-- need to use this for fistlayer
-- to ensure that it is bounded contents by its own length

type RawBIntList n = RawPList n BoundedInt

type BIntList m n = PList m n BoundedInt

-- have to rewrite RawPList to avoid defunctionalisation of;
--type RawBIntMatrix n = RawPList n RawBIntList 
type RawBIntMatrix n = [RawBIntList n]
-- can still take length of this (is prev layer length)

type BIntMatrix m n = List m (RawBIntList n)
-- = List m (RawPList n BoundedInt)
-- = List m [BoundedInt n]

fromBIntMatrix :: (IsList (List n (RawBIntList m))) => BIntMatrix n m -> [[Int]]
fromBIntMatrix x = map (map fromBoundedInt) (fromList x)


-- first layer is a list containing lists of bounded ints
-- which are bounded to be as long as this list
-- the inner lists are of lengths 1 and,
-- each of the ints these contain,
-- are the same as their position in the outer list.
-- however, as none of the other layers have the length of the list (the number of ingoing edges) bounded 
-- (they should!? cant have more edges than the length of the preceeding list...)
-- would it also be useful to be able to bound the max edges eg, for covnets? anticipating adding and removing edges...
-- this is more advanced than currently nesacary...
type FirstLayer n = BIntMatrix n n 

-- have to initialise first layer to have only one edge, to the horizontally adjacent input, 
-- note, the scaffold does not have weights, but the net does, and these should be 1 for the input layer
-- FUCK this would have the activation not being tanh!! maybe its not too bad to scale everything uniformly by this first sigmoid...
-- no
-- make an abstraction so that the activation function can be suppled to the evaluation of one layer to the next
-- this this, with id for the activation function, can be used for applying the input

firstScaffoldLayer :: (IsList (BIntMatrix n n),IsNat n) => Proxy (n :: Nat) -> FirstLayer n
firstScaffoldLayer p = toList p (take (fromNat (getNat p)) (map (return.toBoundedInt) [0..]))

{-
*Main> firstScaffoldLayer (Proxy @(ToNat 3))
[[BoundedInt 0],[BoundedInt 1],[BoundedInt 2]]
-}

{-
at the moment
the edges are only encoded at value level
this can be changed by using singleton Ints
then, the edge connectivity can be encoded at type level
and a proceadure at type level can be used to construct the net
this poses problems for pruning schemes
since the type of the net would change
but this is already the case since nodes cannot be deleted.
this could be solved by allowing the type of the net to change.
then, it *might* be faster that by doing this, the ints are determined from the type level computation
however, that whole concepr relies on the pruning scheme being determined at compile time
which prevents dynamic pruning
this motivates training in batches where compilation is used to change the nets shape
however, it is unlikely that the type level computations at compile time will be faster than at runtime
creating a problem for fast online net reshaping
-}





data Snocs a where
 EmptySnoc :: Snocs a
 SnocOne :: a -> Snocs a
 Snoc :: Snocs a -> a -> Snocs a

type ToSnocs (xs :: [a]) = ToSnocs' xs EmptySnoc 

type family ToSnocs' (xs :: [a]) (ys :: Snocs a) :: Snocs a where
 ToSnocs' '[] ys = ys
 ToSnocs' (x ': xs) ys = Snoc (ToSnocs' xs ys) x

type FromSnocs (xs :: Snocs a) = FromSnocs' xs '[] 

type family FromSnocs' (xs :: Snocs a) (ys :: [a]) :: [a] where
 FromSnocs' EmptySnoc ys = ys
 FromSnocs' (Snoc xs x) ys = x ': (FromSnocs' xs ys)

-- SCAFFOLD-- SCAFFOLD-- SCAFFOLD-- SCAFFOLD-- SCAFFOLD-- SCAFFOLD-- SCAFFOLD-- SCAFFOLD-- SCAFFOLD-- SCAFFOLD-- SCAFFOLD

-- first layer is connected to an input vector 
-- which serves as the previous layer to be accessed
-- but that has connections only between horizontally adjacent nodes
-- this gives them bounds the length of this list
data Scaffold (ns :: Snocs Nat) where
 InitScaffold :: FirstLayer n -> Scaffold (SnocOne n)
 ScaffoldSnoc :: Scaffold (Snoc ns n1) -> BIntMatrix n2 n1 -> Scaffold (Snoc (Snoc ns n1) n2)

unScaffoldSnoc :: forall ns n1 n2. Scaffold (Snoc (Snoc ns n1) n2) -> (Scaffold (Snoc ns n1),BIntMatrix n2 n1)
unScaffoldSnoc (ScaffoldSnoc xs x) = (xs,x) 
-- first layer is at the deepest tail!? yes as it is the one to initialise...
{-
ok, there is some strangeness with the cons list
the overall neural net is a regular vector thing
which actually is order agnostic
but if it were a list
then it would be of a mapAccumForm,
where the foldl or foldr would handle the carry of the preceeding row,
infact, the current layer is stored, and used to calculate the next in a consuming fold
!?!?
this is really fucked,
since the whole net is eventually inhabited
this would take far more memory than the fold..
well, perhaps not, since the Double of the caluclated activation might not be too much compared to the weight Doubles
even though it can be deallocated after having been used to calculate the next layer. 
considerations such as, will the large matrix lookup be slower than the nyaming of the heads in a fold
ie, the structured consumption of the mapaccum computation could use less memory. 
this is essentially the same rationalle behind tail recursion as oopsed to repeatedly traversing a memory block to subsequent positions. 
-}

-- this serves to perform the bounds checks on converting to BoundedInt
-- is reverse for the snoces aspect, combined with the bounds checking over previous layer (automatic from the scaffold constructor)

{-
ToSnocs lengths ~ 'Snoc ns n
IsNat n,
createScaffold'
          :: Columns lengths [Int]
             -> FirstLayer n -> Scaffold ('Snoc ('Snoc ns n) n)
-}
--type 
type CreateScaffoldConstraints n ns lengths =  () :: Constraint
{-
 (IsList (List n [BoundedInt n])
 ,GetNats lengths
 ,ToSnocs lengths ~ 'Snoc ns n
 ,IsNat n
 ) 
-}

type family MashSnocs (snocs :: Snocs a) (xs :: [a]) :: Snocs a where
 MashSnocs snocs '[] = snocs
 MashSnocs snocs (x ': xs) = MashSnocs (Snoc snocs x) xs


class 
    CreateScaffold            snocs lengths where
 createScaffold' :: Scaffold snocs
                 -> Columns lengths [Int]
                 -> Scaffold (MashSnocs snocs lengths)

instance 
 (IsList (List n [Int])
 ,GetNats xs
 ,CreateScaffold ('Snoc ('Snoc sn s) n) xs
 ,IsNat s
 ,Functor (List n)
 ) 
 =>CreateScaffold ('Snoc sn s) (n ': xs) where
 createScaffold' scaffoldSoFar cs = createScaffold' x cs'  -- performs bounds check
  where
   x :: Scaffold ('Snoc ('Snoc sn s) n)
   x = (ScaffoldSnoc y z)
   z :: BIntMatrix n s
   z = (fmap (map toBoundedInt) c)
   y :: Scaffold ('Snoc sn s)
   y = scaffoldSoFar 
   (c,cs') = unconsColumns cs

instance CreateScaffold snocs '[] where
 createScaffold' scaffoldSoFar (Columns []) = scaffoldSoFar 

createScaffold 
 :: (CreateScaffold ('SnocOne n) lengths
    ,IsList (List n [BoundedInt n])
    ,IsNat n
    )
 => Proxy n
 -> Columns lengths [Int]
 -> Scaffold (MashSnocs ('SnocOne n) lengths)
createScaffold p cs = createScaffold' ( InitScaffold (firstScaffoldLayer p)) cs

fromScaffold :: forall lengths. FromScaffold lengths => Scaffold (lengths :: Snocs Nat) -> [[[Int]]]
fromScaffold xs = reverse (fromScaffold' @lengths xs)
 where

class FromScaffold (lengths :: Snocs Nat) where
  fromScaffold' :: Scaffold lengths -> [[[Int]]]

instance (IsList (List n [BoundedInt n]),IsNat n) => FromScaffold (SnocOne n) where
 fromScaffold' (InitScaffold x) = [fromBIntMatrix x]

instance 
 (IsList (List n2 [BoundedInt n1])
 ,IsList (List n [BoundedInt n])
 ,FromScaffold ('Snoc ns n1)) 
 => FromScaffold (Snoc (Snoc ns n1) n2) where
 fromScaffold' (s ) = (fromBIntMatrix x) : (fromScaffold' xs)
  where
   (xs,x) = unScaffoldSnoc @ns @n1 @n2 s

performScaffoldBoundsCheck
  :: -- forall {n :: Nat} {lengths :: [Nat]}.
     (FromScaffold (MashSnocs ('SnocOne n) lengths),
      CreateScaffold ('SnocOne n) lengths, 
      IsList (List n [BoundedInt n])) =>
     Proxy n -> Columns lengths [Int] -> [[[Int]]]
performScaffoldBoundsCheck p cs = fromScaffold $ createScaffold p cs 

--createScaffold = InitScaffold 
--firstScaffoldLayer :: (ToSized [[BoundedInt n]] n,ToUnsized (List n [BoundedInt n]),IsNat n) => Proxy (n :: Nat) -> FirstLayer n

-- a way to construct colums of ints specifying edges
-- with a way to ensure that the length of the lists 
-- provided by the output type
-- is not exceeded by the input ints...
-- thus allowing it to pass the check of createScaffold
-- this needs a function to generate ints, given the index, and the bound, to unfold the list
prepareScaffold :: Int
prepareScaffold = undefined 

-- fuck
-- the bounds checking happens at value level!
-- it is simply done via a smart constructor 
-- to the bounded ints held in the scaffold
-- this is ok, but leverages no real type level mechanism
-- it should not be slow however as it just an initialisation proceadure
-- wait, it does actually use the types to give the shape of the net,
-- and uses this to *set* the bounds checking process for the edge BoundedInt constructors...

-- what is a neural net?
-- does it need the same snoc structure?
-- does it need the bounds?
-- no, the checking is performed when passing into the scaffold
-- the net itself is just columns
-- these columns should be abstracted to use arbitrary Container instances

class (Functor (NetContainer netWrapper), Container (NetContainer netWrapper)) => NeuralNet (netWrapper :: [Nat] -> * -> *) where
 type NetContainer netWrapper :: * -> *
 unwrapNet :: forall container lengths a. container ~ NetContainer netWrapper => (netWrapper lengths a) -> container (container a) -- isnt actually supposed to be used, but must be able to provide it
 netMap :: (a -> b) -> NetContainer netWrapper (NetContainer netWrapper a)
                    -> NetContainer netWrapper (NetContainer netWrapper b)
 netMap f xs = fmap (fmap f) xs
 netWrapperCons :: Int


instance NeuralNet IColumns where
 type NetContainer IColumns = RawIVec 
 unwrapNet (IColumns a) = a
 
--netCube :: [[[Int]]] -> 


-- ALERT
-- not be good maybe to remove the type annotations while casting to net
-- oh, cant avoid, as has VEC
-- but can wrap all to retain? + phantom

data Net (netWrapper :: [Nat] -> * -> *) (lengths :: [Nat]) where
 Net :: (NeuralNet netWrapper,container ~ NetContainer netWrapper) => container (container (Int,Double)) -> Net netWrapper lengths 

toNet :: forall netWrapper lengths. (GetNats lengths,NeuralNet netWrapper) => netWrapper lengths Int -> Net netWrapper lengths 
toNet = Net . netMap @netWrapper  (\(i::Int) -> (i,0::Double)) .  unwrapNet 

-- initialise the net with weights to 0,
-- a simple mapping pairing the edge Int with the weight Double
{-
initialiseNet :: forall netWrapper lengths. NeuralNet netWrapper => Scaffold (ToSnocs lengths) -> Net netWrapper lengths 
-- cast from scaffold, having performed bounds check, 
initialiseNet (ScaffoldSnoc xs x) = initialiseNet xs -- toNet 
initialiseNet (InitScaffold x) = undefined
-}
-- creates a fully connected net
createNet :: (GetNats lengths,NeuralNet netWrapper) => Proxy (lengths :: [Nat]) -> Net netWrapper lengths
createNet = undefined
