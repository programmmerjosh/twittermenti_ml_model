import Cocoa
import CreateML

//let mySS = "iOS13\\ &\\ Swift\\ 5\\ -\\ The\\ Complete\\ iOS\\ App\\ Dev\\ Bootcamp"
let urlString = "/Users/Josh2015Mac/Downloads/Courseware/iOS13&Swift5-TheCompleteiOSAppDevBootcamp/Twittermenti-iOS13-master/twitter-sanders-apple3.csv"

let data = try MLDataTable(contentsOf: URL(fileURLWithPath: urlString))

let(trainingData, testingData) = data.randomSplit(by: 0.8, seed: 5)

let sentimentClassifier = try MLTextClassifier(trainingData: trainingData, textColumn: "text", labelColumn: "class")

let evaluationMetrics = sentimentClassifier.evaluation(on: trainingData, textColumn: "text", labelColumn: "class")

let evaluationAccuracy = (1.0 - evaluationMetrics.classificationError) * 100

let metadata = MLModelMetadata(author: "Joshua van Niekerk", shortDescription: "A model trained to classify sentiment on Tweets", version: "1.0")

try sentimentClassifier.write(to: URL(fileURLWithPath: "/Users/Josh2015Mac/Downloads/Courseware/iOS13&Swift5-TheCompleteiOSAppDevBootcamp"))

try sentimentClassifier.prediction(from: "@Apple is a terrible company!")

try sentimentClassifier.prediction(from: "I just found the best restuarant ever, and it's @DuckandWaffle")

try sentimentClassifier.prediction(from: "I think @CocaCola ads are just ok.")
