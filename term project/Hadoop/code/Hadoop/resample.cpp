extern "C"
void Resample(const float *MovingImage,
						float *ForwardImage,
						const float *DisplaceFieldx,
					    const float *DisplaceFieldy,
					    const float *DisplaceFieldz,
		                unsigned Dsizex,
		                unsigned Dsizey,
		                unsigned Dsizez,
		                unsigned ImageSizex,
		                unsigned ImageSizey,
		                unsigned ImageSizez,
						float ForwardImageOriginx,
						float ForwardImageOriginy,
						float ForwardImageOriginz,
						float MovingImageOriginx,
						float MovingImageOriginy,
						float MovingImageOriginz,
						float maxValue,
						float minValue,
		                float scale)
{

	float *gpuDFx, *gpuDFy, *gpuDFz;
	float *gpuMovingImage, *gpuForwardImage;
	unsigned int Dsize = Dsizex*Dsizey*Dsizez;
	unsigned int Isize = ImageSizex*ImageSizey*ImageSizez;

	cudaMalloc((void **)&gpuDFx, sizeof(float)*Dsize);
	cudaMalloc((void **)&gpuDFy, sizeof(float)*Dsize);
	cudaMalloc((void **)&gpuDFz, sizeof(float)*Dsize);
	cudaMemcpy(gpuDFx, DisplaceFieldx, sizeof(float)*Dsize, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuDFy, DisplaceFieldy, sizeof(float)*Dsize, cudaMemcpyHostToDevice);
	cudaMemcpy(gpuDFz, DisplaceFieldz, sizeof(float)*Dsize, cudaMemcpyHostToDevice);


	cudaMalloc((void **)&gpuMovingImage, sizeof(float)*Isize);
	cudaMalloc((void **)&gpuForwardImage, sizeof(float)*Isize);
	cudaMemcpy(gpuMovingImage, MovingImage, sizeof(float)*Isize, cudaMemcpyHostToDevice);

	int uint = 8;
	dim3 grid((ImageSizex+uint-1)/uint,(ImageSizey+uint-1)/uint,(ImageSizez+uint-1)/uint);
	dim3 block(uint,uint,uint);

	TransformPoint<<<grid, block>>>(gpuForwardImage,
									gpuMovingImage,
									gpuDFx,
									gpuDFy,
									gpuDFz,
									ImageSizex,
									ImageSizey,
									ImageSizez,
									Dsizex,
									Dsizey,
									Dsizez,
									ForwardImageOriginx,
									ForwardImageOriginy,
									ForwardImageOriginz,
									MovingImageOriginx,
								    MovingImageOriginy,
									MovingImageOriginz,
									maxValue,
									minValue,
									scale);

	cudaMemcpy(ForwardImage, gpuForwardImage, sizeof(float)*Isize, cudaMemcpyDeviceToHost);
	cudaFree(gpuDFx);
	cudaFree(gpuDFy);
	cudaFree(gpuDFz);
	cudaFree(gpuMovingImage);
	cudaFree(gpuForwardImage);
}






__global__ void TransformPoint(float *OutputImage,
									float *InputImage,
									const float *DisplaceFieldx,
									const float *DisplaceFieldy,
									const float *DisplaceFieldz,
									int ImageSizex,
									int ImageSizey,
									int ImageSizez,
									int xsize,
									int ysize,
									int zsize,
									float ForwardImageOriginx,
									float ForwardImageOriginy,
									float ForwardImageOriginz,
									float MovingImageOriginx,
									float MovingImageOriginy,
									float MovingImageOriginz,
									float maxValue,
									float minValue,
									float scale)
{
	//transform the point from index to physical
#define MovingImageDimension 3
	unsigned idx = blockIdx.x*blockDim.x+threadIdx.x;
	unsigned idy = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned idz = blockIdx.z*blockDim.z+threadIdx.z;
	if(idx<ImageSizex&&idy<ImageSizey&&idz<ImageSizez)
	{
			float PhysicalePoint[3];
			//transform the index to the fixed image space by ysing fixed image origin, i.e. forwardImgeOrigin.
			PhysicalePoint[0] = idx + ForwardImageOriginx;
			PhysicalePoint[1] = idy + ForwardImageOriginy;
			PhysicalePoint[2] = idz + ForwardImageOriginz;
			const int m_Neighbors = 1 << MovingImageDimension;
			int baseIndex[MovingImageDimension];
			float distance[MovingImageDimension];
			int neighIndex[MovingImageDimension];
			float StartContinuousIndex[MovingImageDimension] = {-0.5/scale + ForwardImageOriginx, -0.5/scale + ForwardImageOriginy, -0.5/scale + ForwardImageOriginz};
			float EndContinuousIndex[MovingImageDimension];
			float EndIndex[MovingImageDimension];
			float index[MovingImageDimension];
			double DisplaceValue[3] = {0,0,0};

			int EndImageIndex[MovingImageDimension];
			int StartImageIndex[MovingImageDimension];
			float ImageStartContinuousIndex[MovingImageDimension];
			float ImageEndContinuousIndex[MovingImageDimension];
			double index_Im[MovingImageDimension];

			//interpolation the image
			ImageEndContinuousIndex[0] = ImageSizex - 0.5;
			ImageEndContinuousIndex[1] = ImageSizey - 0.5;
			ImageEndContinuousIndex[2] = ImageSizez - 0.5;
			EndImageIndex[0] = ImageSizex - 1;
			EndImageIndex[1] = ImageSizey - 1;
			EndImageIndex[2] = ImageSizez - 1;
			for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
			{
				ImageStartContinuousIndex[dim] = -0.5;
				StartImageIndex[dim] = 0;
			}


			EndContinuousIndex[0] = (xsize-0.5)/scale + ForwardImageOriginx;
			EndContinuousIndex[1] = (ysize-0.5)/scale + ForwardImageOriginy;
			EndContinuousIndex[2] = (zsize-0.5)/scale + ForwardImageOriginz;
			EndIndex[0] = xsize -1;
			EndIndex[1] = ysize -1;
			EndIndex[2] = zsize -1;

            index[0] = PhysicalePoint[0];
            index[1] = PhysicalePoint[1];
            index[2] = PhysicalePoint[2];

			bool IsInsideBuffer = true;
			for (unsigned int dim = 0; dim<MovingImageDimension; dim++)
			{
				if(index[dim]<StartContinuousIndex[dim] || index[dim]>EndContinuousIndex[dim])
				{
					IsInsideBuffer = false;
				}
			}

			float StartIndex[MovingImageDimension] = {0,0,0};
			if(IsInsideBuffer)
			{
				index[0] = (index[0] - ForwardImageOriginx)*scale;
				index[1] = (index[1] - ForwardImageOriginy)*scale;
				index[2] = (index[2] - ForwardImageOriginz)*scale;
				//index[1] = 0;
				//index[2] = 0;
				for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
				{
					baseIndex[dim] = floor(index[dim]);
					distance[dim] = index[dim] - baseIndex[dim];
				}

				float totalOverlap = 0;
				for ( unsigned int counter = 0; counter < m_Neighbors; ++counter )
				{
					float overlap = 1.0;    // fraction overlap
					unsigned int upper = counter;  // each bit indicates upper/lower neighbour
					for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
					{
						neighIndex[dim] = baseIndex[dim];
					}
					for ( unsigned int dim = 0; dim < MovingImageDimension; ++dim )
					{
						if ( upper & 1 )
						{

							++(neighIndex[dim]);
						   if ( neighIndex[dim] > EndIndex[dim] )
						   {
							   neighIndex[dim] = EndIndex[dim];
						   }
						   overlap *= distance[dim];
						}
						else
						{
						   if ( neighIndex[dim] < StartIndex[dim] )
						   {
							 neighIndex[dim] = StartIndex[dim];
						   }
						   overlap *= 1.0 - distance[dim];
						 }

						upper >>= 1;
					}

					if(overlap)
					{
						DisplaceValue[0] += DisplaceFieldx[neighIndex[2]*xsize*ysize +  neighIndex[1]*xsize + neighIndex[0]] * overlap;
						DisplaceValue[1] += DisplaceFieldy[neighIndex[2]*xsize*ysize +  neighIndex[1]*xsize + neighIndex[0]] * overlap;
						DisplaceValue[2] += DisplaceFieldz[neighIndex[2]*xsize*ysize +  neighIndex[1]*xsize + neighIndex[0]] * overlap;
						totalOverlap += overlap;
					}
					if (totalOverlap == 1.0)
					{
						break;
					}
				}
			}
			//xtransformed[idx] = DisplaceValue[0];
			//index_Im[0] = double(idx) + DisplaceValue[0];
			//index_Im[1] = double(idy) + DisplaceValue[1];
			//index_Im[2] = double(idz) + DisplaceValue[2];

			//all of the transform are done in the physical space.
			index_Im[0] = PhysicalePoint[0] + DisplaceValue[0];
			index_Im[1] = PhysicalePoint[1] + DisplaceValue[1];
			index_Im[2] = PhysicalePoint[2] + DisplaceValue[2];

			//the function of following codes is transform the physical point to the Moving continuouse index spacing by using the moving image origin.
			index_Im[0] -= MovingImageOriginx;
			index_Im[1] -= MovingImageOriginy;
			index_Im[2] -= MovingImageOriginz;

			bool IsInsideBufferIm = true;
			float ImageValue = 0;
			double nindex[MovingImageDimension];
			for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
			{
				if(index_Im[dim] < ImageStartContinuousIndex[dim] || index_Im[dim] > ImageEndContinuousIndex[dim])
					IsInsideBufferIm = false;
			}
			if (!IsInsideBufferIm)
			{
				for ( unsigned int j = 0; j < MovingImageDimension; j++ )
				{
				  nindex[j] = index_Im[j];
				  float size = ImageEndContinuousIndex[j] - ImageStartContinuousIndex[j];
				  while(nindex[j] > EndImageIndex[j])
				  {
					nindex[j] -= size;
				  }
				  while(nindex[j] < StartImageIndex[j])
				  {
					nindex[j] += size;
				  }
				}
			}
			else
			{
				for ( unsigned int j = 0; j < MovingImageDimension; j++ )
				{
				  nindex[j] = index_Im[j];
				}
			}

			for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
			{
				baseIndex[dim] = floor(nindex[dim]);
				distance[dim] = nindex[dim] - baseIndex[dim];
			}

			float totalOverlap = 0;
			for ( unsigned int counter = 0; counter < m_Neighbors; ++counter )
			{
				float overlap = 1.0;    // fraction overlap
				unsigned int upper = counter;  // each bit indicates upper/lower neighbour
				for (unsigned int dim = 0; dim < MovingImageDimension; ++dim)
				{
					neighIndex[dim] = baseIndex[dim];
				}
				for ( unsigned int dim = 0; dim < MovingImageDimension; ++dim )
				{
					if ( upper & 1 )
					{
						++(neighIndex[dim]);
					   if ( neighIndex[dim] > EndImageIndex[dim] )
					   {
						   neighIndex[dim] = EndImageIndex[dim];
					   }
					   overlap *= distance[dim];
					}
					else
					{
					   // Take care of the case where the pixel is just
					   // in the outer lower boundary of the image grid.
					   if ( neighIndex[dim] < StartIndex[dim] )
					   {
						 neighIndex[dim] = StartIndex[dim];
					   }
					   overlap *= 1.0 - distance[dim];
					 }

					upper >>= 1;
				}
				if(overlap)
				{
					ImageValue += InputImage[neighIndex[2]*ImageSizex*ImageSizey +  neighIndex[1]*ImageSizex + neighIndex[0]] * overlap;
					totalOverlap += overlap;
				}
				if (totalOverlap == 1.0)
				{
					break;
				}
			}
			if (ImageValue>maxValue) ImageValue = maxValue;
			if (ImageValue<minValue) ImageValue = minValue;
			OutputImage[idz*ImageSizex*ImageSizey+idy*ImageSizex+idx] = ImageValue;
	}
}


