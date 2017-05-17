package mapred.ngramcount;

import java.io.IOException;

import mapred.util.Tokenizer;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.conf.Configuration;

public class NgramCountMapper extends Mapper<LongWritable, Text, Text, NullWritable> {

	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		Configuration conf = context.getConfiguration();
		int arg = conf.getInt("n", 1);
		String line = value.toString();
		String[] words = Tokenizer.tokenize(line);

		for (int i = 0; i < words.length-arg+1; i++) {
			StringBuilder bld = new StringBuilder();

			for (int n = 1; n <= arg && n <= words.length - i; n++) {
				if(n > 1) bld.append(" ");
				bld.append(words[i-1+n]);
			}
			Text ng = new Text(bld.toString());
			context.write(ng, NullWritable.get());
		}
		/*
		for (String word : words)
			context.write(new Text(word), NullWritable.get());
		*/

	}
}
