package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import mapred.util.InputLines;
import mapred.util.FileUtil;
import mapred.filesystem.CommonFileOperations;

import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.DefaultStringifier;

public class SimilarityMapper extends Mapper<LongWritable, Text, IntWritable, Text> {

	Map<String, Map<String, Integer>> allfeatures = new HashMap<String, Map<String, Integer>>();

	/**
	 * We compute the inner product of feature vector of every hashtag with that
	 * of #job
	 */
	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String[] hashtag_featureVector = line.split("\\s+", 2);

		String hashtag = hashtag_featureVector[0];
		Map<String, Integer> features = parseFeatureVector(hashtag_featureVector[1]);

		for (String tag : allfeatures.keySet()) {
			if(hashtag.compareTo(tag) < 0) {
				Integer similarity = computeInnerProduct(allfeatures.get(tag), features);
				if(similarity.intValue() > 0)
					context.write(new IntWritable(similarity), new Text(hashtag + "\t" + tag));
			}
		}
	}

	/**
	 * This function is ran before the mapper actually starts processing the
	 * records, so we can use it to setup the job feature vector.
	 * 
	 * Loads the feature vector for hashtag #job into mapper's memory
	 */
	@Override
	protected void setup(Context context) 
		throws IOException, InterruptedException {
		
		List<String> list = new ArrayList<String>();

		String dir = context.getConfiguration().get("input");

		String[] all_dir = CommonFileOperations.listAllFiles(dir, null, false);

		for(String file_dir : all_dir) {
			InputLines allfeatureVector = FileUtil.loadLines(file_dir);
			for(String lines : allfeatureVector)
				list.add(lines);
		}

		for(String allline : list) {
			String[] inputtag_featureVector = allline.split("\\s+", 2);
			String tags = inputtag_featureVector[0];
			Map<String, Integer> tagFeatures = parseFeatureVector(inputtag_featureVector[1]);
			allfeatures.put(tags, tagFeatures);
		}
	}

	/**
	 * De-serialize the feature vector into a map
	 * 
	 * @param featureVector
	 *            The format is "word1:count1;word2:count2;...;wordN:countN;"
	 * @return A HashMap, with key being each word and value being the count.
	 */
	private Map<String, Integer> parseFeatureVector(String featureVector) {
		Map<String, Integer> featureMap = new HashMap<String, Integer>();
		String[] features = featureVector.split(";");
		for (String feature : features) {
			String[] word_count = feature.split(":");
			featureMap.put(word_count[0], Integer.parseInt(word_count[1]));
		}
		return featureMap;
	}

	/**
	 * Computes the dot product of two feature vectors
	 * @param featureVector1
	 * @param featureVector2
	 * @return 
	 */
	private Integer computeInnerProduct(Map<String, Integer> featureVector1,
			Map<String, Integer> featureVector2) {
		Integer sum = 0;
		for (String word : featureVector1.keySet()) 
			if (featureVector2.containsKey(word))
				sum += featureVector1.get(word) * featureVector2.get(word);
		
		return sum;
	}
}














