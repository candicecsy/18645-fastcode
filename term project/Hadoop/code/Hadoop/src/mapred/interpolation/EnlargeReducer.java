package mapred.interpolation;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

public class EnlargeReducer extends Reducer<Text, Text, Text, Text> {

	@Override
	protected void reduce(Text key, Iterable<Text> value, Context context)
			throws IOException, InterruptedException {
				List<String> list = new ArrayList<String>();

				for (Text word : value) {
					String w = word.toString();
					list.add(w);
				}

				double size = list.size();
				double proportion = 1/size;

				int r = 0,g = 0,b = 0;
				for(String line : list){
					String[] rgb = line.split("\\s+");	

					Integer r1 = Integer.parseInt(rgb[2]);
					Integer g1 = Integer.parseInt(rgb[3]);
					Integer b1 = Integer.parseInt(rgb[4]);
					int rcopy = r1.intValue();
					int gcopy = g1.intValue();
					int bcopy = b1.intValue();
					r += (int)(rcopy * proportion);
					g += (int)(gcopy * proportion);
					b += (int)(bcopy * proportion); 
				}

				context.write(key, new Text(Integer.toString(r) + "\t" + 
					Integer.toString(g) + "\t" + Integer.toString(b)));

			}
}