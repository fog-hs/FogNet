{-# Language ScopedTypeVariables #-}
import System.Random
import Control.Monad

import RawIVec
import Container

egSize = 8 -- 262,144.0

type V = RawIVec Double

type Neuron = (Int,V)

neuron :: [Double] -> Neuron 
neuron xs = (length xs,toRawIVec xs)

initializeInput :: Neuron
initializeInput = initializeNeuron 1

initializeNeuron :: Int -> Neuron
initializeNeuron i = neuron $ replicate i 0

runNeuron :: V -> Neuron -> Double
runNeuron ys (i,xs) | length ys == i = tanh $ sum $ zipWithC (*) (xs) (ys)
 | otherwise = error", runtime nets!! HLIST!!!!" 

testNeuron = runNeuron (toRawIVec [1,1,0]) $ neuron [0.1,0.11,0.2]

{-
*Main> testNeuron
0.2069664997294526
-}

type Layer = RawIVec Neuron
type Net = RawIVec Layer

type ProtoNet = [[Neuron]]

createNet' :: [Int] -> ProtoNet 
createNet' (x:xs) = (inputLayer x) : (createNet' x xs)
 where
  createNet' :: Int -> [Int] -> ProtoNet 
  createNet' i (x:[]) = [createLayer i x]
  createNet' i (x:xs) = (createLayer i x) : (createNet' x xs)
  inputLayer :: Int -> [Neuron]
  inputLayer i = replicate i initializeInput 
  createLayer :: Int -> Int -> [Neuron]
  createLayer i j = replicate j $ initializeNeuron i

createNet :: [Int] -> RawIVec (RawIVec Neuron)
createNet xs = toRawIVec $ map (toRawIVec) $ createNet' xs

runLayer :: V -> Layer -> V
runLayer xs ns = zipWithC runNeuron (toRawIVec (replicate (length ns) xs)) ns


-- here is the main difference
-- dont want to write as in fold...
runNet :: V -> Net -> V
runNet xs n = let v = zipWithC runNeuron (mapC return xs) y 
                   in runNet' v ys
 where
  (y:ys) = fromRawIVec n
  runNet' :: V -> [Layer] -> V
  runNet' xs [] = xs
  runNet' xs (y:ys) = runNet' (runLayer xs y) ys

egNet :: Net
egNet = createNet (replicate 32 32)

test1 :: V
test1 = runNet (toRawIVec (replicate 32 1)) egNet 
	
--
{-
updateNet :: (V -> V) -> Net -> Net
updateNet f = map (map (\(a,b) -> (a,f b)))

test2 :: Bool
test2 = runNet (replicate 32 1) (updateNet (map (+ 0.1)) egNet) == (replicate 32 0.9966097505839319)

simpleAutoEncoder :: (Int,Int,Int,Int,Int) -> Net
simpleAutoEncoder x@(inputN,encoderLen,bottleneckN,decoderN,outputN) = createNet (aoutoencoderGeometry x)
 where
  aoutoencoderGeometry (a,b,c,d,e) = interpolate (a,b,c) ++ interpolate (c,d,e)

interpolate :: (Int,Int,Int) -> [Int]
interpolate (a,b,c) = map f [0..b-1]
 where
  f :: Int -> Int
  f x = floor $ (\a -> a/(fromIntegral (b-1))) $ fromIntegral $ ((b-1)-x)*a + x*c

{-
*Main> interpolate (10,5,5)
[10,8,7,6,5]
-}
autoTuple x = let y = div x 2 in (x,y,y,y,x)
egAutoEncoder n = simpleAutoEncoder $ autoTuple n 

test3 :: Bool
--test3 = runNet (replicate 32 1) (updateNet (map (+ 0.1)) egAutoEncoder) == (replicate 32 0.9948642936392333)
test3 = (egSize ==) $ length $ runNet (replicate egSize 1) (updateNet (map (+ 0.1)) (egAutoEncoder egSize)) 


----
-- now try to learn 
-- id autoencoder for sin waves

sinOf n = map sin [0,((2*pi)/(fromIntegral (n-1))) .. (2*pi)]

trial1 :: V
trial1 = runNet (sinOf egSize) (updateNet (map (+ 0.001)) (egAutoEncoder egSize)) 
 where

test4 = (length trial1) == egSize

updateNetIO :: (V -> IO V) -> Net -> IO Net
updateNetIO f net = traverse (traverse (g f)) net
 where
  g :: (V -> IO V) -> Neuron -> IO Neuron
  g f = (\(a,b) -> ((,) a) <$> (f b))

randomWeight :: Double -> IO Double
randomWeight x = randomRIO (0-x,x)

stochasticNetUpdate :: Double -> Net -> IO Net
stochasticNetUpdate w net = updateNetIO (traverse (f w)) net
 where
  f :: Double -> Double -> IO Double
  f w x = (x+) <$> randomWeight w


-- simple monte carlo
-- accept a random update if better
firstTraining :: Int -> Net -> IO Net
firstTraining 0 x = return x
firstTraining n x = firstTraining (n-1) =<< metropolis input target update x
 where 
  input = target
  target = sinOf egSize
  update :: Net -> IO Net
  update = stochasticNetUpdate (0.1)

l2 xs ys = sum $ map (^2) $ zipWith (-) xs ys

metropolis :: [Double] -> [Double] -> (Net -> IO Net) -> Net -> IO Net
metropolis input target update net = do
 let e1 = l2 input $ runNet input net
 net' <- update net
 let e2 = l2 input $ runNet input net'
-- really should accept some backwards steps...
 if e2 <= e1 then return net' else return net


go1 n = do
 let input = sinOf egSize
 let net0 = egAutoEncoder egSize
 let o1 = runNet input net0
 let e1 = l2 input o1
 net' <- firstTraining n net0
 let o2 = runNet input net'
 let e2 = l2 input o2
 print e1
 print e2
-}
