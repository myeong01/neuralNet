package main

import (
	"encoding/csv"
	"fmt"
	"neuralNet/neuralnet"
	"os"
	"strconv"
)

func main() {
	n := neuralnet.New(784, 100, 10, 0.1, nil)

	trainFile, err := os.Open("./mnist_train.csv")
	if err != nil {
		panic(err)
	}
	defer trainFile.Close()
	trainSetReader := csv.NewReader(trainFile)
	rawTrainSet, err := trainSetReader.ReadAll()
	if err != nil {
		panic(err)
	}
	trainSet := make([][]float64, len(rawTrainSet))
	for yIndex := range trainSet {
		trainSet[yIndex] = make([]float64, len(rawTrainSet[yIndex]))
		for xIndex := range trainSet[yIndex] {
			trainSet[yIndex][xIndex], err = strconv.ParseFloat(rawTrainSet[yIndex][xIndex], 64)
			if err != nil {
				panic(err)
			}
			if xIndex != 0 {
				trainSet[yIndex][xIndex] = (trainSet[yIndex][xIndex] / 255 * 0.99) + 0.01
			}
		}
	}
	fmt.Println("Start Train")
	epoch := 5
	for i := 0; i < epoch; i++ {
		fmt.Println("epoch", i, "start")
		for _, train := range trainSet {
			target := make([]float64, 10)
			for index := range target {
				target[index] = 0.01
			}
			target[int(train[0])] = 0.99
			err := n.Train(train[1:], target)
			if err != nil {
				panic(err)
			}
		}
	}
	fmt.Println("Finish Train")

	testFile, err := os.Open("./mnist_test.csv")
	if err != nil {
		panic(err)
	}
	defer testFile.Close()
	testSetReader := csv.NewReader(testFile)
	rawTestSet, err := testSetReader.ReadAll()
	if err != nil {
		panic(err)
	}
	testSet := make([][]float64, len(rawTestSet))
	for yIndex := range testSet {
		testSet[yIndex] = make([]float64, len(rawTestSet[yIndex]))
		for xIndex := range testSet[yIndex] {
			testSet[yIndex][xIndex], err = strconv.ParseFloat(rawTestSet[yIndex][xIndex], 64)
			if err != nil {
				panic(err)
			}
			if xIndex != 0 {
				testSet[yIndex][xIndex] = (testSet[yIndex][xIndex] / 255 * 0.99) + 0.01
			}
		}
	}
	var correctCnt int64 = 0
	for _, test := range testSet {
		answer, err := n.Query(test[1:])
		if err != nil {
			panic(err)
		}
		if int(test[0]) == findMaxIndex(answer) {
			correctCnt++
		}
	}
	fmt.Println("Accuracy :", float64(correctCnt)/float64(len(testSet)))
}

func findMaxIndex(l []float64) int {
	maxValue := l[0]
	maxIndex := 0
	for index, value := range l {
		if maxValue < value {
			maxValue = value
			maxIndex = index
		}
	}
	return maxIndex
}
