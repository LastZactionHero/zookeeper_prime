import code

def print_results(X, Y, model):
    print "Evaluating Model..."
    correct = 0
    incorrect = 0
    total = len(Y)

    for idx, y in enumerate(Y):
        print "%d / %d" % (idx, total)

        x = X[idx]
        prediction = model.predict([x])[0]
        predictionIdx = prediction.index(max(prediction))
        actualIdx = y.index(max(y))

        print predictionIdx
        print actualIdx

        if(predictionIdx == actualIdx):
            print "  - correct"
            correct += 1
        else:
            print prediction
            print y
            print "  - incorrect"
            incorrect += 1

    percentCorrect = float(correct) / float(total)
    print "Correct:   %d" % correct
    print "Incorrect: %d" % incorrect
    print "Percent:   %f" % percentCorrect
    print "Total:     %d" % total
