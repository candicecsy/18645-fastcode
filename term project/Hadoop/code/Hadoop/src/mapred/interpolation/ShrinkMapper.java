package mapred.interpolation;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class ShrinkMapper extends Mapper<LongWritable, Text, Text, Text> {

	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
				String line = value.toString();
				String[] words = line.split("\\s+");

				Integer row = Integer.parseInt(words[0]);
				Integer colume = Integer.parseInt(words[1]);

				int x = row.intValue();
				int y = colume.intValue();

				if(x%2 == 0 && y%2 == 0){
					context.write(new Text(Integer.toString(x/2) + "\t" + Integer.toString(y/2)),
						new Text(words[2] + "\t" + words[3] + "\t" + words[4]));
				}

			}
}