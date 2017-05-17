package mapred.interpolation;

import java.io.IOException;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class TranslationMapper extends Mapper<LongWritable, Text, Text, Text> {


	String[] feature;

	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String[] parameter = line.split("\\s+");

		Integer row = Integer.parseInt(parameter[0]);
		Integer colume = Integer.parseInt(parameter[1]);

		int x = row.intValue();
		int y = colume.intValue();

		Integer size_row = Integer.parseInt(feature[0]);
	    Integer size_colume = Integer.parseInt(feature[1]);

	    int sr = size_row.intValue();
	    int sc = size_colume.intValue();

		/*
		 * map out the translated data
		 */

		if((x+100) <= sr && (y+200) <= sc) {
			context.write(new Text(Integer.toString(x+100) + "\t" + Integer.toString(y+200)),
				new Text(parameter[2] + "\t" + parameter[3] + "\t" + parameter[4]));
		

			if(x == 1 && y == 1) {
				for(int i = 1; i < x + 100; i++) {
					for(int j = 1; j < y + 200; j++) {
						context.write(new Text(Integer.toString(i) + "\t" + Integer.toString(j)),
							new Text(Integer.toString(255) + "\t" + Integer.toString(255) + "\t" 
								+ Integer.toString(255)));
					}
				}
			}

			if(x == 1){
				for(int i = 1; i < x + 100; i++){
					context.write(new Text(Integer.toString(i) + "\t" + Integer.toString(y+200)),
							new Text(Integer.toString(255) + "\t" + Integer.toString(255) + "\t" 
								+ Integer.toString(255)));
				}
			}

			if(y == 1){
				for(int i = 1; i < y + 200; i++){
					context.write(new Text(Integer.toString(x+100) + "\t" + Integer.toString(i)),
							new Text(Integer.toString(255) + "\t" + Integer.toString(255) + "\t" 
								+ Integer.toString(255)));
				}
			}

		}
	}

	@Override
	protected void setup(Context context)
			throws IOException, InterruptedException {

			String size = context.getConfiguration().get("inputsize");
			feature = size.split("\\s+");
	}
}