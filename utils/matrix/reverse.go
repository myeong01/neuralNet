package matrix

func Reverse(a [][]float64) [][]float64 {
	b := make([][]float64, len(a[0]))
	for yIndex := range b {
		b[yIndex] = make([]float64, len(a))
		for xIndex := range b[yIndex] {
			b[yIndex][xIndex] = a[xIndex][yIndex]
		}
	}
	return b
}
