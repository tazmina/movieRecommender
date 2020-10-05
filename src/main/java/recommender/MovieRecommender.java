package recommender;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.mllib.recommendation.ALS;
import org.apache.spark.mllib.recommendation.MatrixFactorizationModel;
import org.apache.spark.mllib.recommendation.Rating;
import org.apache.spark.SparkConf;

import java.util.List;

/**
 * Collaborative filtering
 *
 * Usage:
 *  /opt/mapr/spark/spark-2.1.0/bin/spark-submit
 *  --class recommender.MovieRecommender
 *  CS185-jar-with-dependencies.jar
 *  <rating_file>
 *  <movie_file>
 *  <user_to_query_file>
 */
public class MovieRecommender {

    public static void main(String[] args) {

        if (args.length < 3) {
            System.err.println("Usage: MovieRecommender <rating_file> <movie_file> <user_to_query_file>");
            System.err.println("eg: MovieRecommender rating.data movie.data user_to_query.data");
            System.exit(1);
        }

        String ratingPath = args[0];
        String moviePath = args[1];
        String userToQueryPath = args[2];

        SparkConf conf = new SparkConf().setAppName("MovieRecommender");
        JavaSparkContext jsc = new JavaSparkContext("local[2]", "MovieRecommender", conf);

        // Load and parse the rating data
        JavaRDD<String> ratingRdd = jsc.textFile(ratingPath);  // load rating file
        JavaRDD<Rating> ratings = ratingRdd.map(s -> {
            String[] sarray = s.split(",");
            return new Rating(Integer.parseInt(sarray[0]),  // user
                    Integer.parseInt(sarray[1]),            // movie
                    Double.parseDouble(sarray[2]));         // rating
        });

        // Index movie data
        JavaRDD<String> movieRdd = jsc.textFile(moviePath);  // load movie file
        JavaRDD<Tuple2<Integer, String>> movies = movieRdd.map(m -> {
            String[] sarray = m.split(",");
            return new Tuple2<>(
                    Integer.parseInt(sarray[0]),  // movie id
                    sarray[1]);                   // movie name
        });

        // Index query users
        JavaRDD<Integer> users = jsc.textFile(userToQueryPath)
                .map(Integer::parseInt);          // query user id

        // Build the recommendation model using ALS
        int rank = 8;
        int numIterations = 10;

        MatrixFactorizationModel model = ALS.train(
            JavaRDD.toRDD(ratings),
            rank,
            numIterations,
            0.01
        );

        System.out.println("Model has been trained" +
                ", num ratings: " + ratings.collect().size() +
                ", num movies: " + movies.collect().size());

        // Find unseen movies for the user
        JavaRDD<Tuple2<Object, Object>> userAllMovies =
                users.cartesian(movies.map(Tuple2::_1))
                .map(r -> new Tuple2<>(r._1(), r._2()));

        JavaRDD<Tuple2<Object, Object>> userSeenMovies = ratings
                .map(r -> new Tuple2<>(r.user(), r.product()));

        JavaRDD<Tuple2<Object, Object>> userUnseenMovies = userAllMovies.subtract(userSeenMovies);

        // Find the predictions
        List<Tuple2<Tuple2<Integer, Integer>, Double>> predictions =
            model.predict(JavaRDD.toRDD(userUnseenMovies)).toJavaRDD()
                    .map(r -> new Tuple2<>(new Tuple2<>(r.user(), r.product()), r.rating()))
                    .sortBy(Tuple2::_2, false, 10)
                    .take(10);

        // Final step, find movie name
        predictions.forEach(r -> {
            int movieId = r._1._2;
            String movieName = movies.filter(m -> m._1 == movieId).first()._2();  // find movie name
            System.out.println("User: " + r._1._1 + ", Movie: " + movieName + ", Rating: " + r._2);
        });

        jsc.stop();
    }
}
