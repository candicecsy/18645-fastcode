package mapred.interpolation;

import java.io.IOException;

import mapred.util.InputLines;
import mapred.util.FileUtil;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.fs.Path;

public class EnlargeMapper extends Mapper<LongWritable, Text, Text, Text> {

	String[] feature;

	@Override
	protected void map(LongWritable key, Text value, Context context)
	         throws IOException, InterruptedException {
	    String line = value.toString();
	    String[] parameter = line.split("\\s+");

	    Integer size_row = Integer.parseInt(feature[0]);
	    Integer size_colume = Integer.parseInt(feature[1]);

	    int sr = size_row.intValue();
	    int sc = size_colume.intValue();

	    Integer row = Integer.parseInt(parameter[0]);
	    Integer colume = Integer.parseInt(parameter[1]);

	    int x = row.intValue();
	    int y = colume.intValue();

	    //store the point self
	    context.write(new Text(Integer.toString(2*x) + "\t" + Integer.toString(2*y)), value);

	    if(y != sc) {
	    	//store the point y + 0.5
	    	context.write(new Text(Integer.toString(2*x) + "\t" + Integer.toString((int)(2*(y+0.5)))),
	    		value);

	    	//store the point x - 0.5, y + 0.5
	    	context.write(new Text(Integer.toString((int)(2*(x-0.5))) + "\t" + Integer.toString((int)(2*(y+0.5)))),
	    		value);
	    }

	    if(x != sr) {
	    	//store the point, x + 0.5
	    	context.write(new Text(Integer.toString((int)(2*(x+0.5))) + "\t" + Integer.toString(2*y)),
	    		value);	

	    	//store the point x + 0.5, y - 0.5
	    	context.write(new Text(Integer.toString((int)(2*(x+0.5))) + "\t" + Integer.toString((int)(2*(y-0.5)))),
	    	value);
	    }

	    if(y != sc && x != sr) {
	    	//store the point x + 0.5, y + 0.5
	    	context.write(new Text(Integer.toString((int)(2*(x+0.5))) + "\t" + Integer.toString((int)(2*(y+0.5)))),
	    		value);
		}

	    //store the point x, y - 0.5
	   	context.write(new Text(Integer.toString(2*x) + "\t" + Integer.toString((int)(2*(y-0.5)))),
	    	value);
	   
	   	//store the point x-0.5, y
	   	context.write(new Text(Integer.toString((int)(2*(x-0.5))) + "\t" + Integer.toString(2*y)),
	    	value);
	    
	   	//store the point x - 0.5, y - 0.5
	   	context.write(new Text(Integer.toString((int)(2*(x-0.5))) + "\t" + Integer.toString((int)(2*(y-0.5)))),
	    	value);
	    

	}

	@Override
	protected void setup(Context context)
			throws IOException, InterruptedException {

			String size = context.getConfiguration().get("inputsize");
			feature = size.split("\\s+");
		}
}