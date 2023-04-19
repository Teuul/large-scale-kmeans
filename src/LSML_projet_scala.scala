// Databricks notebook source
// MAGIC %md
// MAGIC **DATA**

// COMMAND ----------

val raw_iris: RDD[String] = sc.textFile("/FileStore/tables/iris_data.txt")

val raw_beans: RDD[String] = sc.textFile("/FileStore/tables/Dry_Bean_Dataset-2.csv")

// COMMAND ----------

def get_labels(raw_rdd: RDD[String]) : Array[String] = {
  /*
    Returns data labels
  */
  raw_rdd
  .map(str => str.split(",").last)
  .distinct().collect()
}

def clean_data(raw_rdd: RDD[String]) : (RDD[(Long, Array[Double])], RDD[(Long, String)]) = {
  /*
    Formats data for clustering models
  */
  val cleaned = raw_rdd
  .map(str => str.split(","))
  .map({case(a: Array[String]) => {(a.dropRight(1).map( _.toDouble), a.last)} })
  .zipWithIndex()
  .map(x => (x._2, x._1))
  
  (
    cleaned.map(x => (x._1, x._2._1)),
    cleaned.map(x => (x._1, x._2._2))
  )
}

// COMMAND ----------

// Data variables for testing
val aux = clean_data(raw_beans)

val data : RDD[(Long, Array[Double])] = aux._1
val classification: RDD[(Long, String)] = aux._2

// COMMAND ----------

// Perimeter attribute distribution
val perimeter = data.map({case(i, xi) => xi(1)}).toDF()
perimeter.createOrReplaceTempView("perimeter")
display(spark.sql("select * from perimeter"))

// COMMAND ----------

// Solidity attribute distribution
val solidity = data.map({case(i, xi) => xi(10)}).toDF()
solidity.createOrReplaceTempView("solidity")
display(spark.sql("select * from solidity"))

// COMMAND ----------

// MAGIC %md
// MAGIC **CODEBASE**

// COMMAND ----------

def distance(P1: Array[Double], P2: Array[Double]): Double = {
  scala.math.sqrt((P1 zip P2).map(x => scala.math.pow((x._1 - x._2), 2)).reduce((x,y) => x+y))
}

distance(data.take(2)(0)._2, data.take(2)(1)._2)

// COMMAND ----------

def sum(P1: Array[Double], P2: Array[Double]): Array[Double] = {
  (P1 zip P2).map(x => x._1 + x._2)
}

sum(data.take(2)(0)._2, data.take(2)(1)._2)

// COMMAND ----------

def substract(P1: Array[Double], P2: Array[Double]): Array[Double] = {
  (P1 zip P2).map(x => x._1 - x._2)
}

substract(Array(3.0, 2.5), Array(1.0, 0.5))

// COMMAND ----------

def divide(P1: Array[Double], P2: Array[Double]): Array[Double] = {
  (P1 zip P2).map(x => x._1 / x._2)
}

divide(Array(3.0, 2.5), Array(3.0, 0.5))

// COMMAND ----------

def mult_vect(k: Double, P: Array[Double]): Array[Double] = {
  P.map(k * _)
}

mult_vect(1.0/5.0, data.take(1)(0)._2)

// COMMAND ----------

def init_centroids(data: RDD[(Long, Array[Double])], k: Int) : Array[(Long, Array[Double])] = {
  /*
    Naive centroids initialization
  */
  data.takeSample(false, k, System.nanoTime.toInt)
}

init_centroids(data, 7).map(_._1)

// COMMAND ----------

// MAGIC %md
// MAGIC Weighted sampling package from: https://gist.github.com/Ichoran/24df1ca6d6616e1bd34f4013b1018f89

// COMMAND ----------

def cuml(wt: Array[Double]) = {
  val l = wt.length
  val base = if ((l & (l-1)) == 0) l else java.lang.Integer.highestOneBit(l)*2
  val tree = new Array[Double](base*2)
  System.arraycopy(wt, 0, tree, 0, wt.length)
  var in = 0
  var out = base
  var n = base
  while (in + 1 < out) {
    while (in + 1 < n) {
      tree(out) = tree(in) + tree(in+1)
      out += 1
      in += 2
    }
    n = n | (n >>> 1)
  }
  tree
}

def seek(tree: Array[Double], p: Double)(zero: Int = tree.length-4, index: Int = 0, stride: Int = 2): Int = {
  if (zero == 0) index + (if (p < tree(index)) 0 else 1)
  else if (p < tree(zero + index)) seek(tree, p)(zero - (stride << 1), index << 1, stride << 1)
  else seek(tree, p - tree(zero + index))(zero - (stride << 1), (index << 1) + 2, stride << 1)
}

def wipe(tree: Array[Double], index: Int)(value: Double = tree(index), width: Int = tree.length >> 1) {
  tree(index) -= value
  if (width > 1) wipe(tree, (tree.length+index) >> 1)(value, width >> 1)
}

// Random number generator should generate values in (0, 1]
def sample(r: () => Double, wt: Array[Double], k: Int): Array[Int] = {
  val indices = new Array[Int](k)
  val tree = cuml(wt)
  var i = 0
  while (i < k) {
    val index = seek(tree, r()*tree(tree.length-2))()
    wipe(tree, index)()
    indices(i) = index
    i += 1
  }
  indices
}

// COMMAND ----------

def nearest(P: Array[Double], centroids: Array[Array[Double]]): Int = {
  /*
    Nearest centroid from given point
  */
  var min_index: Int = -1
  var minimum: Double = Double.MaxValue
  for (i <- (0 until centroids.length)) {
    var centroid = centroids(i)
    var d = distance(P, centroid)
    if (d < minimum) {
      minimum = d
      min_index = i
    }
  }
  min_index
}

nearest(data.take(1)(0)._2, init_centroids(data, 7).map(x => x._2))

// COMMAND ----------

def init_centroids_plusplus(data: RDD[(Long, Array[Double])], k: Int) : Array[(Long, Array[Double])] = {
  /*
    Smart random centroids initialization (kmeans++)
  */
  val r = scala.util.Random
  val firstDraw : (Long, Array[Double])= (0, data.takeSample(false, 1, System.nanoTime.toInt)(0)._2)
  var centroids: Array[(Long, Array[Double])] = Array[(Long, Array[Double])]()
  centroids = centroids ++ Array(firstDraw)
  
  for (i <- (1 until k)) {
    val aux : RDD[(Long, Double)] = data.map({case(id, x) => (id, distance(x, centroids(nearest(x, centroids.map(_._2)))._2))})
    val weight_tuples : Array[(Long, Double)]= aux.collect()
    val tuple_index = sample(r.nextDouble, weight_tuples.map(_._2), 1)(0)
    val centroid_id = weight_tuples(tuple_index)._1
    val new_centroid : (Long, Array[Double]) = (i.toLong, data.filter(t => t._1 == centroid_id).take(1)(0)._2)
    centroids = centroids ++ Array(new_centroid)
  }
  centroids
}

init_centroids_plusplus(data, 7).map(_._1)

// COMMAND ----------

def get_z(data: RDD[(Long, Array[Double])], centroids: Array[(Long, Array[Double])]) = {
  data
    .map(P => (P._1, nearest(P._2, centroids.map(x => x._2) )))
}

get_z(data, init_centroids(data, 7)).map(x => (x._2, x._1)).countByKey()

// COMMAND ----------

def get_j(data: RDD[(Long, Array[Double])], z: RDD[(Long, Int)], centroids: Array[(Long, Array[Double])]): Double = {
  z
    .join(data)
    .map(e => distance(e._2._2, centroids(e._2._1)._2))
    .reduce((x,y) => x+y)
}

var c_test = init_centroids(data, 7)
var z_test = get_z(data, c_test)
get_j(data, z_test, c_test)

// COMMAND ----------

def compute_centroids(data: RDD[(Long, Array[Double])], z: RDD[(Long, Int)]): Array[(Long, Array[Double])] = {
  z
    .join(data)
    .map({case(i, (k, xi)) => (k, (xi, 1))})
    .reduceByKey({case(((x1, q1), (x2, q2))) => (sum(x1, x2), q1+q2)})
    .map({case(k, (xk, qk)) => (k.toLong, mult_vect(1.toDouble/qk.toDouble, xk))})
    .collect()
}

var z_test = get_z(data, init_centroids(data, 7))
compute_centroids(data, z_test).map(_._1)

// COMMAND ----------

def k_means(data: RDD[(Long, Array[Double])], k: Int) = {
  /*
    Naive implementation of kmeans
  */
  val MAX_ITER: Int = 40
  val EPS: Double = 0.01
  
  var iter: Int = 0
  var centroids: Array[(Long, Array[Double])] = init_centroids(data, k)
  var z: RDD[(Long, Int)] = get_z(data, centroids)
  var j: Double = get_j(data, z, centroids)
  var delta: Double = Double.MaxValue
  var aux: Double = j
  
 
  while(iter < MAX_ITER && delta > EPS) {    
    centroids = compute_centroids(data, z)
    z = get_z(data, centroids)
    j = get_j(data, z, centroids)
    delta = (j/aux - 1).abs
    aux = j
    iter += 1
    
  }
  z
}

def k_means_plusplus(data: RDD[(Long, Array[Double])], k: Int) : RDD[(Long, Int)] = {
  /*
    Kmeans++ implementation: smart random centroids initialization
  */
  val MAX_ITER: Int = 40
  val EPS: Double = 0.01
  
  var iter: Int = 0
  var centroids: Array[(Long, Array[Double])] = init_centroids_plusplus(data, k)
  var z: RDD[(Long, Int)] = get_z(data, centroids)
  var j: Double = get_j(data, z, centroids)
  var delta: Double = Double.MaxValue
  var aux: Double = j
   
  while(iter < MAX_ITER && delta > EPS) {    
    centroids = compute_centroids(data, z)
    z = get_z(data, centroids)
    j = get_j(data, z, centroids)
    delta = (j/aux - 1).abs
    aux = j
    iter += 1
  }
  z
}

// COMMAND ----------

case class ClusterEntry(prediction: Long, classification: String, quantity: Int)

def display_results(prediction: RDD[(Long, Int)], classification: RDD[(Long, String)]) {
  val clustering_data = prediction.join(classification).map(_._2).map(t => ClusterEntry(t._1, t._2, 1)).toDF()
  clustering_data.createOrReplaceTempView("clustering_results")
  display(spark.sql("select * from clustering_results"))
}

// COMMAND ----------

def get_min_row(rdd: RDD[(Array[Double])]) : Array[Double] = {
  /*
    Returns the minimum per each column
  */
  rdd.reduce((x,y) => (x zip y).map(t => if (t._1 <= t._2) t._1 else t._2))
}

def get_max_row(rdd: RDD[(Array[Double])]) : Array[Double] = {
  /*
    Returns the maximum per each column
  */
  rdd.reduce((x,y) => (x zip y).map(t => if (t._1 >= t._2) t._1 else t._2))
}

def normalize_data(rdd: RDD[(Long, Array[Double])]) : RDD[(Long, Array[Double])] = {
  /*
    Normalize data to a scale of 0 to 1, according to attributes minimum and maximum
  */
  val MINS : Array[Double] = get_min_row(rdd.map(_._2))
  val MAXS : Array[Double] = get_max_row(rdd.map(_._2))
  rdd.map({case(id, data) => {(id, divide(substract(data, MINS), substract(MAXS, MINS)))}})
}

// COMMAND ----------

normalize_data(data).take(1)(0)

// COMMAND ----------

// MAGIC %md
// MAGIC Clustering quality metrics

// COMMAND ----------

def davies_bouldin(data: RDD[(Long, Array[Double])], z: RDD[(Long, Int)]) : Double = {
  /*
    Clustering coherence Davies-Bouldin metric
  */
  val centroids : RDD[(Int, Array[Double])] = z
    .join(data)
    .map({case(i, (k, xi)) => (k, (xi, 1))})
    .reduceByKey({case(((x1, q1), (x2, q2))) => (sum(x1, x2), q1+q2)})
    .map({case(k, (xk, qk)) => (k, mult_vect(1.toDouble/qk.toDouble, xk))})
  
  val cluster_distances : RDD[(Int, Double)] = z
    .join(data)
    .map({case(i, (k, xi)) => (k, xi)})
    .join(centroids)
    .map({case(k, (xi, ck)) => (k, (distance(xi, ck), 1))})
    .reduceByKey({case((d1, q1), (d2, q2)) => (d1 + d2, q1 + q2)})
    .map({case(k, (d, q)) => (k, d.toDouble/q.toDouble)})
  
  val aux : (Double, Int) = centroids
    .join(cluster_distances)
    .cartesian(centroids.join(cluster_distances))
    .filter({case((k1 , (ck1, dk1)), (k2 , (ck2, dk2))) => k1 != k2})
    .map({case((k1 , (ck1, dk1)), (k2 , (ck2, dk2))) => (k1, (dk1 + dk2)/distance(ck1, ck2))})
    .reduceByKey({case(x,y) => if (x >= y) x else y})
    .map({case(k, mk) => (mk, 1)})
    .reduce({case((mk1, q1), (mk2, q2)) => (mk1+mk2, q1+q2)})
  (aux._1.toDouble / aux._2.toDouble)
}

// COMMAND ----------

def max(a: Double, b: Double) : Double = {
  if (a >=b ) a else b
}

def silhouette(data: RDD[(Long, Array[Double])], z: RDD[(Long, Int)]) : RDD[(Long, Double)] = {
  /*
    Clustering coherence Silhouette metric
  */
  val a : RDD[(Long, Double)] = z
    .join(data)
    .map({case(i, (k, xi)) => (k, (i, xi))})
    .join(z.join(data).map({case(j, (k, xj)) => (k, (j, xj))}))
    .map({case(k, ((i, xi),(j, xj))) => (i, (distance(xi, xj), 1))})
    .reduceByKey({case((d1, q1), (d2, q2)) => (d1+d2, q1+q2)})
    .map({case(i, (d, size_ck)) => (i, d/(size_ck-1))})
  
  val b : RDD[(Long, Double)] = data.join(z)
  .cartesian(data.join(z))
  .map({case((i,(xi, k)), (j,(xj, l))) => (i, (k,l), (xi,xj))})
  .filter({case(i, (k,l), (xi,xj)) => k!=l})
  .map({case(i, (k,l), (xi,xj)) => ((i,k), (distance(xi, xj), 1))})
  .reduceByKey({case((d1, q1), (d2, q2)) => (d1+d2, q1+q2)})
  .map({case((i,k), (d, q)) => (i, d/q.toDouble)})
  .reduceByKey({case(mk1, mk2) => if (mk1<mk2) mk1 else mk2})
  
  a.join(b).map({case(i, (ai, bi)) => (i, (bi - ai)/max(ai, bi))})
}

def silhouette_coefficient(si: RDD[(Long, Double)]) : Double = {
  /*
    Silhouette coeficient
  */
  val aux = si.map({case(i, s) => (s, 1)}).reduce({case((s1, q1), (s2, q2)) => (s1+s2, q1+q2)})
  (aux._1 / aux._2.toDouble)
}

var test_data = clean_data(raw_iris)._1
var z = get_z(test_data, init_centroids(test_data, 7))

silhouette_coefficient(silhouette(test_data, z))

// COMMAND ----------

// MAGIC %md
// MAGIC **BEANS MODEL**

// COMMAND ----------

val aux = clean_data(raw_beans)
val beans_data = aux._1
val beans_classification = aux._2

// COMMAND ----------

val beans_prediction = k_means(normalize_data(beans_data), 7)

// COMMAND ----------

display_results(beans_prediction, beans_classification)

// COMMAND ----------

// Indice de Davies-Bouldin et Coefficient de Silhouette
print(davies_bouldin(beans_data, beans_prediction))
print(silhouette_coefficient(silhouette(beans_data, beans_prediction)))

// COMMAND ----------

val beans_prediction = k_means_plusplus(normalize_data(beans_data), 7)

// COMMAND ----------

display_results(beans_prediction, beans_classification)

// COMMAND ----------

// Indice de Davies-Bouldin:
davies_bouldin(normalize(beans_data, beans_prediction)

// COMMAND ----------

// MAGIC %md
// MAGIC **IRIS MODEL**

// COMMAND ----------

val aux = clean_data(raw_iris)
val iris_data = aux._1
val iris_classification = aux._2

// COMMAND ----------

val iris_prediction = k_means(normalize_data(iris_data), 3)

// COMMAND ----------

display_results(iris_prediction, iris_classification)

// COMMAND ----------

val iris_prediction = k_means_plusplus(normalize_data(iris_data), 3)

// COMMAND ----------

display_results(iris_prediction, iris_classification)

// COMMAND ----------

// MAGIC %md
// MAGIC **PERFORMANCE ANALYSIS**

// COMMAND ----------

// BEANS: NAIVE MODEL/RAW DATA

var total_time : Long = 0.toLong
var total_score : Double = 0.toDouble
for (i <- (0 until 20)) {
  var data = beans_data
  val start = System.nanoTime()
  val prediction = k_means(data, 7)
  val end = System.nanoTime()
  val exectime = end - start
  total_time = total_time + exectime
  val score = davies_bouldin(data, prediction)
  total_score = total_score + score
}
println("exec time (mean): " + (total_time / 20.toLong) * scala.math.pow(10, -9) + " s")
println("davies-bouldin score (mean): " + total_score/20.toDouble)

// COMMAND ----------

// BEANS: NAIVE MODEL/NORMALIZED DATA

var total_time : Long = 0.toLong
var total_score : Double = 0.toDouble
for (i <- (0 until 20)) {
  var data = normalize_data(beans_data)
  val start = System.nanoTime()
  val prediction = k_means(data, 7)
  val end = System.nanoTime()
  val exectime = end - start
  total_time = total_time + exectime
  val score = davies_bouldin(data, prediction)
  total_score = total_score + score
}
println("exec time (mean): " + (total_time / 20.toLong) * scala.math.pow(10, -9) + " s")
println("davies-bouldin score (mean): " + total_score/20.toDouble)

// COMMAND ----------

// BEANS: KMEANS++ MODEL/NORMALIZED DATA

var total_time : Long = 0.toLong
var total_score : Double = 0.toDouble
for (i <- (0 until 20)) {
  var data = normalize_data(beans_data)
  val start = System.nanoTime()
  val prediction = k_means_plusplus(data, 7)
  val end = System.nanoTime()
  val exectime = end - start
  total_time = total_time + exectime
  val score = davies_bouldin(data, prediction)
  total_score = total_score + score
}
println("exec time (mean): " + (total_time / 20.toLong) * scala.math.pow(10, -9) + " s")
println("davies-bouldin score (mean): " + total_score/20.toDouble)

// COMMAND ----------

// IRIS: NAIVE MODEL/RAW DATA

var total_time : Long = 0.toLong
var total_score : Double = 0.toDouble
for (i <- (0 until 20)) {
  var data = iris_data
  val start = System.nanoTime()
  val prediction = k_means(data, 3)
  val end = System.nanoTime()
  val exectime = end - start
  total_time = total_time + exectime
  val score = davies_bouldin(data, prediction)
  total_score = total_score + score
}
println("exec time (mean): " + (total_time / 20.toLong) * scala.math.pow(10, -9) + " s")
println("davies-bouldin score (mean): " + total_score/20.toDouble)

// COMMAND ----------

// IRIS: NAIVE MODEL/NORMALIZED DATA

var total_time : Long = 0.toLong
var total_score : Double = 0.toDouble
for (i <- (0 until 20)) {
  var data = normalize_data(iris_data)
  val start = System.nanoTime()
  val prediction = k_means(data, 3)
  val end = System.nanoTime()
  val exectime = end - start
  total_time = total_time + exectime
  val score = davies_bouldin(data, prediction)
  total_score = total_score + score
}
println("exec time (mean): " + (total_time / 20.toLong) * scala.math.pow(10, -9) + " s")
println("davies-bouldin score (mean): " + total_score/20.toDouble)

// COMMAND ----------

// IRIS: KMEANS++ MODEL/NORMALIZED DATA

var total_time : Long = 0.toLong
var total_score : Double = 0.toDouble
for (i <- (0 until 20)) {
  var data = normalize_data(iris_data)
  val start = System.nanoTime()
  val prediction = k_means_plusplus(data, 3)
  val end = System.nanoTime()
  val exectime = end - start
  total_time = total_time + exectime
  val score = davies_bouldin(data, prediction)
  total_score = total_score + score
}
println("exec time (mean): " + (total_time / 20.toLong) * scala.math.pow(10, -9) + " s")
println("davies-bouldin score (mean): " + total_score/20.toDouble)
