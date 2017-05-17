package mapred.interpolation;

import java.io.IOException;
import mapred.job.Optimizedjob;
import mapred.util.FileUtil;
import mapred.util.InputLines;
import mapred.util.SimpleParser;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.DefaultStringifier;

public class Driver {
	public static void main(String[] args) throws Exception{
		SimpleParser parser = new SimpleParser(args);

		String input = parser.get("input");
		String outputTranslation = parser.get("outputTranslation");
		String outputEnlarge = parser.get("outputEnlarge");
		String outputShrink = parser.get("outputShrink");
		String inputsize = parser.get("inputsize");
		
		interpolationTranslation(input, outputTranslation, inputsize);

		interpolationEnlarge(input, outputEnlarge, inputsize);

		interpolationShrink(input, outputShrink);
	}

	
	/**
	 * Using the matrix vector, we can compute the interpolation 
	 * of translation.
	 *
	 * @param input
	 * @param output
	 * @throws IOException
	 * @throws ClassNotFoundException
	 * @throws InterruptedException
	 *
	 */
	private static void interpolationTranslation(String input, String output, 
						String inputsize)
		throws IOException, ClassNotFoundException, InterruptedException {


			Configuration conf = new Configuration();
			conf.set("inputsize", inputsize);

			Optimizedjob job = new Optimizedjob(conf, input, output,
						"compute the interpolationTranslation");
			job.setClasses(TranslationMapper.class, null, null);
			job.setMapOutputClasses(Text.class, Text.class);
			job.run();
		}

	/**
	 * Using the matrix vector, we can compute the interpolation 
	 * of enlarge.
	 *
	 * @param input
	 * @param output
	 * @throws IOException
	 * @throws ClassNotFoundException
	 * @throws InterruptedException
	 *
	 */
	private static void interpolationEnlarge(String input, String output, 
					String inputsize)
		throws IOException, ClassNotFoundException, InterruptedException {

			
			Configuration conf = new Configuration();
			conf.set("inputsize", inputsize);

			Optimizedjob job = new Optimizedjob(conf, input, output,
						"compute the interpolationEnlarge");
			job.setClasses(EnlargeMapper.class, EnlargeReducer.class, null);
			job.setMapOutputClasses(Text.class, Text.class);
			job.run();
		}

	/**
	 * Using the matrix vector, we can compute the interpolation 
	 * of Shrink.
	 *
	 * @param input
	 * @param output
	 * @throws IOException
	 * @throws ClassNotFoundException
	 * @throws InterruptedException
	 *
	 */
	private static void interpolationShrink(String input, String output)
		throws IOException, ClassNotFoundException, InterruptedException {

			Optimizedjob job = new Optimizedjob(new Configuration(), input, output,
						"compute the interpolationShrink");
			job.setClasses(ShrinkMapper.class, null, null);
			job.setMapOutputClasses(Text.class, Text.class);
			job.run();
		}
}