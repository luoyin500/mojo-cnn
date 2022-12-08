#pragma once

#include <string>
#include <sstream>

#include "core_math.h"
#include "activation.h"

namespace mojo
{
//#ifdef _WIN32
//#include <windows.h>
//#endif
	/*
	double PCFreq = 0.0;
	__int64 CounterStart = 0;

	void StartCounter()
	{
		LARGE_INTEGER li;
		if (!QueryPerformanceFrequency(&li)) return;
		PCFreq = double(li.QuadPart) / 1000.0;
		QueryPerformanceCounter(&li);
		CounterStart = li.QuadPart;
	}
	double GetCounter()
	{
		LARGE_INTEGER li;
		QueryPerformanceCounter(&li);
		return double(li.QuadPart - CounterStart) / PCFreq;
	}
	*/

#define int2str(a) std::to_string((long long)a)
#define float2str(a) std::to_string((long double)a)
#define bail(txt) {std::cerr << "ERROR :"  << txt << " @ " << __FILE__ <<  ": line " << __LINE__ <<  ": function " << __FUNCTION__  ; throw;}


//----------------------------------------------------------------------------------------------------------
// B A S E   L A Y E R
//
// all other layers derived from this
class base_layer 
{
protected:
	bool _has_weights;
	bool _use_bias;
	float _learning_factor;
	int _thread_count;

public:
	activation_function *p_act;
	
	bool has_weights() {return _has_weights;}
	bool use_bias() { return _use_bias; }
	void set_learning_factor(float f=1.0f) {_learning_factor = 1.f;}
	void set_threading(int thread_count) {_thread_count=thread_count; if(_thread_count<1) _thread_count=1;}

	int pad_cols, pad_rows;
	matrix node;
	matrix bias; // this is something that maybe should be in the same class as the weights... but whatever. handled differently for different layers
	
	std::string name;
	// index of W matrix, index of connected layer
	std::vector<std::pair<int,base_layer*>> forward_linked_layers;
#ifndef MOJO_NO_TRAINING
	matrix delta;
	std::vector<std::pair<int,base_layer*>> backward_linked_layers;

	virtual void distribute_delta(base_layer &top, const matrix &w, const int train = 1) =0;
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train =1)=0;
	virtual void update_bias(const matrix &newbias, float alpha) {};

#endif
	virtual void accumulate_signal(const base_layer &top_node, const matrix &w, const int train =0) =0;

	base_layer(const char* layer_name, int _w, int _h=1, int _c=1) : node(_w, _h, _c),  p_act(NULL), name(layer_name), _has_weights(true), pad_cols(0), pad_rows(0), _learning_factor(1.f), _use_bias(false), _thread_count(1)
		#ifndef MOJO_NO_TRAINING
		,delta(_w,_h,_c,NULL,false)
		#endif
	{
	}

	virtual void resize(int _w, int _h=1, int _c=1)
	{
		if (_w<1) _w = 1; if (_h<1) _h = 1; if (_c<1) _c = 1;
		node =matrix(_w,_h,_c);
		if (_use_bias) { bias = matrix(_w, _h, _c); bias.fill(0.); }
		#ifndef MOJO_NO_TRAINING
		delta =matrix(_w,_h,_c,NULL,false);
		#endif
	}
	
	virtual ~base_layer(){if(p_act) delete p_act;}
	virtual int fan_size() {return node.chans*node.rows*node.cols;}

	virtual void activate_nodes()
	{
		if (p_act)
		{
			if(_use_bias)
				//for (int c=0; c<node.chans; c++) 
				{
					//const float b = bias.x[c];
					//float *x= &node.x[c*node.chan_stride];
					p_act->f(node.x,node.size(),bias.x);
				}
			else
				p_act->f(node.x, node.size(), 0);

		}
	}
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		top.forward_linked_layers.push_back(std::make_pair((int)weight_mat_index,this));
		#ifndef MOJO_NO_TRAINING
		backward_linked_layers.push_back(std::make_pair((int)weight_mat_index,&top));
		#endif
		if (_has_weights)
		{
			int rows = node.cols*node.rows*node.chans;
			int cols = top.node.cols*top.node.rows*top.node.chans;
			return new matrix(cols, rows, 1);
		}
		else
			return NULL;	
	}

	//inline float f(float *in, int i, int size, float bias) {return p_act->f(in, i, size, bias);};
	inline float df(float *in, int i, int size) { if (p_act) return p_act->df(in, i, size); else return 1.f; };
	virtual std::string get_config_string() =0;	
};

//----------------------------------------------------------------------------------------------------------
// I N P U T   L A Y E R
//
// input layer class - can be 1D, 2D (c=1), or stacked 2D (c>1)
class input_layer : public base_layer
{
public:
	input_layer(const char *layer_name, int _w, int _h=1, int _c=1) : base_layer(layer_name,_w,_h,_c) {p_act=new_activation_function("identity"); }
	virtual  ~input_layer(){}
	virtual void activate_nodes() { /*node.reset_empty_chans(); */}
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train =1) {}
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train =1) {}
	virtual void accumulate_signal(const base_layer &top_node, const matrix &w, const int train =0) {}
	virtual std::string get_config_string() {std::string str="input "+int2str(node.cols)+" "+int2str(node.rows)+" "+int2str(node.chans)+ " "+p_act->name+"\n"; return str;}
};

//----------------------------------------------------------------------------------------------------------
// F U L L Y   C O N N E C T E D
//
// fully connected layer
class fully_connected_layer : public base_layer
{
public:
	fully_connected_layer(const char *layer_name, int _size, activation_function *p) : base_layer(layer_name, _size, 1, 1) 
	{
		p_act = p; _use_bias = true;	
		bias = matrix(node.cols, node.rows, node.chans);
		bias.fill(0.);

	}//layer_type=fully_connected_type;}
	virtual std::string get_config_string() {std::string str="fully_connected "+int2str(node.size())+ " "+p_act->name+"\n"; return str;}
	virtual void accumulate_signal( const base_layer &top,const matrix &w, const int train =0)
	{
		// doesn't care if shape is not 1D
		// here weights are formated in matrix, top node in cols, bottom node along rows. (note that my top is opposite of traditional understanding)
		// node += top.node.dot_1dx2d(w);
		const int s = w.rows;
		const int ts = top.node.size();
		const int ts2 = top.node.cols*top.node.rows;

		// this can be sped up a little with SSE. 
		if(top.node.chan_stride!=ts2)
		{
			//std::cout << "here: " << top.node.chan_stride << ","<< ts2 << ","<< top.node.chans << ":";
			MOJO_THREAD_THIS_LOOP(_thread_count)
			for (int j = 0; j < s; j++)  
			{
				for (int i = 0; i < top.node.chans; i++) 
				{
					node.x[j] += dot(top.node.x+top.node.chan_stride*i, w.x+j*w.cols+ts2*i, ts2);  
					//float *f=top.node.x+top.node.chan_stride*i;
					//if(node.x[j]!=node.x[j])
					if(node.x[j]!=node.x[j])
					{
						//std::cout << "stuff" << top.name << " " << name << " " << top.node.x[top.node.chan_stride*i] << " " << w.x[j*w.cols+ts2*i] << " | " ;
						for (int k=0; k<top.node.size(); k++)
						{
							std::cout << k<< ","<< top.node.x[k] <<",";
						}
						
						exit(1);
					}
				}
			}
		}
		else
		{
		MOJO_THREAD_THIS_LOOP(_thread_count)
		for (int j = 0; j < s; j++)  node.x[j] += dot(top.node.x, w.x+j*w.cols, ts);  
		}
	}
#ifndef MOJO_NO_TRAINING
	virtual void update_bias(const matrix &newbias, float alpha) {
		for (int j = 0; j < bias.size(); j++) bias.x[j] -= newbias.x[j] * alpha;
	}
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train =1)
	{
		if(top.delta.cols*top.delta.rows==top.delta.chan_stride)
		{
			const int w_cols = w.cols;
			for (int b = 0; b < delta.size(); b++)
			{
				const float cb = delta.x[b];
				for (int t = 0; t < top.delta.size(); t++) 
					top.delta.x[t] += cb*w.x[t + b*w_cols];
			}
		}
		else
		{
			const int w_cols = w.cols;
			const int chan_size=top.delta.cols*top.delta.rows;

			for (int b = 0; b < delta.size(); b++)
			{
				const float cb = delta.x[b];
				for (int tc = 0; tc < top.delta.chans; tc++)	
					for (int t = 0; t < chan_size; t++)	
							top.delta.x[t+tc*top.delta.chan_stride] += cb*w.x[t + tc*chan_size + b*w_cols];
			}

				 
			
		}


	}

	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train = 1)
	{
		const float *bottom = delta.x; const int sizeb = delta.size();
		const float *top = top_layer.node.x; const int sizet = top_layer.node.cols*top_layer.node.rows*top_layer.node.chans;
		dw.resize(sizet, sizeb, 1);

		for (int b = 0; b < sizeb; b++)
		{
			const float cb = bottom[b];
			const int chan_size =  top_layer.node.cols*top_layer.node.rows;
			if(sizet!=top_layer.node.size())
			{
				//std::cout << "calculate_dw - odd size";
				for (int tc = 0; tc < top_layer.node.chans; tc++)	
					for (int t = 0; t < chan_size; t++)	
					{
						dw.x[t+tc*chan_size + b*sizet] = top[t+tc*top_layer.node.chan_stride] * cb;
						//std::cout << dw.x[t+tc*chan_size + b*sizet] <<",";
					}
				 
			}
			else
			{
				for (int t = 0; t < sizet; t++)	dw.x[t + b*sizet] = top[t] * cb;
			}
		}
	}
#endif

};

//----------------------------------------------------------------------------------------------------------
// M A X   P O O L I N G   
// 
// may split to max and ave pool class derived from pooling layer.. but i never use ave pool anymore
class max_pooling_layer : public base_layer
{

protected:
	int _pool_size;
	int _stride;
	// uses a map to connect pooled result to top layer
	std::vector<int> _max_map;
public:
	max_pooling_layer(const char *layer_name, int pool_size) : base_layer(layer_name, 1)
	{
		 _stride = pool_size; _pool_size = pool_size; //layer_type=pool_type;
		_has_weights = false;
	}
	max_pooling_layer(const char *layer_name, int pool_size, int stride ) : base_layer(layer_name, 1)
	{
		_stride= stride; _pool_size=pool_size;  //layer_type=pool_type;
		_has_weights = false;
	}
	virtual  ~max_pooling_layer(){}
	virtual std::string get_config_string() {std::string str="max_pool "+int2str(_pool_size) +" "+ int2str(_stride) +"\n"; return str;}

	// ToDo would like delayed activation of conv layer if available
//	virtual void activate_nodes(){ return;}
	virtual void resize(int _w, int _h=1, int _c=1)
	{
		if(_w<1) _w=1; if(_h<1) _h=1; if(_c<1) _c=1;
		_max_map.resize(_w*_h*_c);
		base_layer::resize(_w, _h, _c);
	}
	// no weights 
	virtual void calculate_dw(const base_layer &top_layer, matrix &dw, const int train =1) {}
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
			// need to set the size of this layer
		// can really only handle one connection comming in to this
		int pool_size = _pool_size;
		int w = (top.node.cols) / pool_size;
		int h = (top.node.rows) / pool_size;
		if (_stride != _pool_size)
		{
			w = 1 + ((top.node.cols - _pool_size) / _stride);
			h = 1 + ((top.node.rows - _pool_size) / _stride);
		}

		resize(w, h, top.node.chans);

		return base_layer::new_connection(top, weight_mat_index);
	}

	// this is downsampling
	// the pool size must fit correctly in the image map (use resize prior to call if this isn't the case)
	virtual void accumulate_signal(const base_layer &top,const matrix &w,const int train =0)
	{
		int kstep = top.node.chan_stride; // top.node.cols*top.node.rows;
		int jstep=top.node.cols;
		int output_index=0;
		int *p_map = _max_map.data();
		int pool_y=_pool_size; if(top.node.rows==1) pool_y=1; //-top.pad_rows*2==1) pool_y=1;
		int pool_x=_pool_size; if(top.node.cols==1) pool_x=1;//-top.pad_cols*2==1) pool_x=1;
		const float *top_node = top.node.x;

		for(int k=0; k<top.node.chans; k++)
		{
			for(int j=0; j<=top.node.rows- _pool_size; j+= _stride)
			{
				for(int i=0; i<=top.node.cols- _pool_size; i+= _stride)
				{
					const int base_index=i+(j)*jstep+k*kstep;
					int max_i=base_index;
					float max=top_node[base_index];
					if(pool_x==2)
					{
						const float *n=top_node+base_index;
						//if(max<n[0]) { max = n[0]; max_i=max_i;}
						if(max<n[1]) { max = n[1]; max_i=base_index+1;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+jstep+1;}
					}
					else if(pool_x==3)
					{
						const float *n=top_node+base_index;
						//if(max<n[0]) { max = n[0]; max_i=max_i;}
						if(max<n[1]) { max = n[1]; max_i=base_index+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+jstep+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+jstep+2;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+2*jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+2*jstep+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2*jstep+2;}
					}
					else if(pool_x==4)
					{
						const float *n=top_node+base_index;
						//if(max<n[0]) { max = n[0]; max_i=max_i;}
						if(max<n[1]) { max = n[1]; max_i=base_index+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+3;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+jstep+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+jstep+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+jstep+3;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+2*jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+2*jstep+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+2*jstep+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+2*jstep+3;}
						n+=jstep;
						if(max<n[0]) { max = n[0]; max_i=base_index+3*jstep;}
						if(max<n[1]) { max = n[1]; max_i=base_index+3*jstep+1;}
						if(max<n[2]) { max = n[2]; max_i=base_index+3*jstep+2;}
						if(max<n[3]) { max = n[3]; max_i=base_index+3*jstep+3;}
					}
					else	
					{
						// speed up with optimized size version
						for(int jj=0; jj<pool_y; jj+= 1)
						{
							for(int ii=0; ii<pool_x; ii+= 1)
							{
								int index=i+ii+(j+jj)*jstep+k*kstep;
								if((max)<(top_node[index]))
								{
									max = top_node[index];
									max_i=index;

								}
							}
						}

					}
					//if (max<1e-5) node.empty_chan[k] = 1;
					//else node.empty_chan[k] = 0;

					node.x[output_index] = top_node[max_i];
					p_map[output_index] = max_i;
					output_index++;
					
				}
			}
		}
	}
#ifndef MOJO_NO_TRAINING

	// this is upsampling
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train =1)
	{
		int *p_map = _max_map.data();
		const int s = (int)_max_map.size();
		for(int k=0; k<s; k++) top.delta.x[p_map[k]]+=delta.x[k];
	}
#endif
};

//----------------------------------------------------------------------------------------------------------
// C O N V O L U T I O N   
//
class convolution_layer : public base_layer
{
	int _stride;
public:
	int kernel_rows;
	int kernel_cols;
	int maps;
	//int maps_per_kernel;
	int kernels_per_map;
	int groups;


	convolution_layer(const char *layer_name, int _w, int _c, int _s, activation_function *p ) : base_layer(layer_name, _w, _w, _c) 
	{
		p_act=p; _stride =_s; kernel_rows=_w; kernel_cols=_w; maps=_c;kernels_per_map=0; pad_cols = kernel_cols-1; pad_rows = kernel_rows-1;
		_use_bias = true;
		groups=1;
	}

	convolution_layer(const char *layer_name, int _w, int _c, int _s, int _g, activation_function *p ) : base_layer(layer_name, _w, _w, _c) 
	{
		p_act=p; _stride =_s; kernel_rows=_w; kernel_cols=_w; maps=_c;kernels_per_map=0; pad_cols = kernel_cols-1; pad_rows = kernel_rows-1;
		_use_bias = true;
		groups=_g;
	}

	virtual  ~convolution_layer() {
	}

	virtual std::string get_config_string() 
	{
		if(groups==1) {std::string str="convolution "+int2str(kernel_cols)+" "+int2str(maps)+" " + int2str(_stride) + " " +p_act->name+"\n"; return str;}
		else {std::string str="group_convolution "+int2str(kernel_cols)+" "+int2str(maps)+" " + int2str(_stride)+" " + int2str(groups) + " " +p_act->name+"\n"; return str;}
	}
	
	virtual int fan_size() { return kernel_rows*kernel_cols*maps*kernels_per_map; }

	
	virtual void resize(int _w, int _h=1, int _c=1) // special resize nodes because bias handled differently with shared wts
	{
		if(kernel_rows*kernel_cols==1) node =matrix(_w,_h,_c);  /// use special channel aligned matrix object
		else node =matrix(_w,_h,_c,NULL,true);  /// use special channel aligned matrix object

		bias =matrix(1,1,_c);
		bias.fill(0.);
		#ifndef MOJO_NO_TRAINING
		if(kernel_rows*kernel_cols==1) delta =matrix(_w,_h,_c);  /// use special channel aligned matrix object
		else delta =matrix(_w,_h,_c,NULL,true);  /// use special channel aligned matrix object
		#endif
	}

	// this connection work won't work with multiple top layers (yet)
	virtual matrix * new_connection(base_layer &top, int weight_mat_index)
	{
		top.forward_linked_layers.push_back(std::make_pair(weight_mat_index,this));
		#ifndef MOJO_NO_TRAINING
		backward_linked_layers.push_back(std::make_pair(weight_mat_index,&top));
		#endif
		// re-shuffle these things so weights of size kernel w,h,kerns - node of size see below
		//int total_kernels=top.node.chans*node.chans;
		kernels_per_map += top.node.chans;
		resize((top.node.cols-kernel_cols)/_stride+1, (top.node.rows-kernel_rows)/_stride+1, maps);

		return new matrix(kernel_cols,kernel_rows, maps*kernels_per_map);
	}

	// activate_nodes
	virtual void activate_nodes()
	{ 
		const int map_size = node.rows*node.cols;
		const int map_stride = node.chan_stride;
		const int _maps = maps;

		MOJO_THREAD_THIS_LOOP(_thread_count)
		for (int c=0; c<_maps; c++) 
		{
			p_act->fc(&node.x[c*map_stride],map_size,bias.x[c]);
			//if(node.x[c*map_stride]!=node.x[c*map_stride]) bail("activate");
			
		}
	}


	virtual void accumulate_signal( const base_layer &top, const matrix &w, const int train =0)
	{	
		const int kstep = top.node.chan_stride;// NOT the same as top.node.cols*top.node.rows;
		const int jstep=top.node.cols;
		//int output_index=0;
		const int kernel_size=kernel_cols*kernel_rows;
		const int kernel_map_step = kernel_size*kernels_per_map;
		const int map_size=node.cols*node.rows;
		const int map_stride = node.chan_stride;

		const float *_w = w.x;
		const int top_chans = top.node.chans;
		const int map_cnt=maps;
		const int w_size = kernel_cols;
		const int stride = _stride;		
		const int node_size= node.cols;
		const int top_node_size = top.node.cols;
		const int outsize = node_size*node_size;
		//printf("%d %d %d,", top.node.chans, node.chans, groups);
		if ((top.node.chans == node.chans) && (top.node.chans==groups))
		{
			//	printf("here");
			if(kernel_rows>=2 && (kernel_rows<=5))
			{
				matrix img_ptr(node_size, node_size, kernel_rows*kernel_rows, NULL, true);

				for (int k = 0; k < map_cnt; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					unwrap_aligned_NxN(kernel_rows, img_ptr.x, &top.node.x[k*kstep], jstep, stride);
					float *ww = &w.x[(0 + k*maps)*kernel_size];

					if(kernel_rows==2)
					{
						//MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
						dotsum_unwrapped_2x2(img_ptr.x, ww+k*kernel_size, node.x + map_stride*k, outsize);
					}
					else if(kernel_rows==3)
					{
						//MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
						dotsum_unwrapped_3x3(img_ptr.x, ww+k*kernel_size, node.x + map_stride*k, outsize);
					}
					else if(kernel_rows==4)
					{
						//MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
						dotsum_unwrapped_4x4(img_ptr.x, ww+k*kernel_size, node.x + map_stride*k, outsize);
					}
					else //(kernel_rows==5)
					{
						//MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
						dotsum_unwrapped_5x5(img_ptr.x, ww+k*kernel_size, node.x + map_stride*k, outsize);
					}
				}
			}
			else if (kernel_rows == 1)
			{

				for (int k = 0; k < map_cnt; k++) // input channels --- same as kernels_per_map - kern for each input
				{
						const float *_top_node = &top.node.x[k*kstep];
							const float cw = w.x[(k + k*maps)*kernel_size];
							const int mapoff = map_size*k;
							for (int j = 0; j < node_size*node_size; j += stride) node.x[j + mapoff] += _top_node[j] * cw;
				}
			}
		
			return;
		}
		
		if(kernel_rows>=2 && (kernel_rows<=5))
		{
			matrix img_ptr(node_size, node_size, kernel_rows*kernel_rows, NULL, true);

			const int maps_per_group = map_cnt/groups;
			const int top_chan_per_group = top_chans/groups;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;

				for (int k = start_k; k < stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					unwrap_aligned_NxN(kernel_rows, img_ptr.x, &top.node.x[k*kstep], jstep, stride);
					float *ww = &w.x[(0 + k*maps)*kernel_size];

					if(kernel_rows==2)
					{
						MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
						for (int map = start_map; map < stop_map; map+=1) dotsum_unwrapped_2x2(img_ptr.x, ww+map*kernel_size, node.x + map_stride*map, outsize);
					}
					else if(kernel_rows==3)
					{
						MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
						for (int map = start_map; map < stop_map; map+=1) dotsum_unwrapped_3x3(img_ptr.x, ww+map*kernel_size, node.x + map_stride*map, outsize);
					}
					else if(kernel_rows==4)
					{
						MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
						for (int map = start_map; map < stop_map; map+=1) dotsum_unwrapped_4x4(img_ptr.x, ww+map*kernel_size, node.x + map_stride*map, outsize);
					}
					else //(kernel_rows==5)
					{
						MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
						for (int map = start_map; map < stop_map; map+=1) dotsum_unwrapped_5x5(img_ptr.x, ww+map*kernel_size, node.x + map_stride*map, outsize);
					}
				}
			}
		}
		else if (kernel_rows == 1)
		{
			const int maps_per_group = map_cnt/groups;
			const int top_chan_per_group = top_chans/groups;
		//	const int group_start=0;
		//	const int group_end= group_start+maps_per_group;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;

				for (int k = start_k; k < stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					const float *_top_node = &top.node.x[k*kstep];

					//MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
					for (int map = start_map; map < stop_map; map++) 
					{
						const float cw = w.x[(map + k*maps)*kernel_size];
						const int mapoff = map_size*map;
						for (int j = 0; j < node_size*node_size; j += stride) node.x[j + mapoff] += _top_node[j] * cw;
					}
				}
			}
		}
		else
		{

			const int maps_per_group = map_cnt/groups;
			const int top_chan_per_group = top_chans/groups;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;


				for(int map=start_map; map<stop_map; map++) // how many maps  maps= node.chans
				{
					for(int k=start_k; k<stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
					{
						MOJO_THREAD_THIS_LOOP_DYNAMIC(_thread_count)
						for(int j=0; j<node_size; j+= stride) // input h 
 							for(int i=0; i<node_size; i+= stride) // intput w
								node.x[i+(j)*node.cols +map_stride*map]+= 
									unwrap_2d_dot(
										&top.node.x[(i)+(j)*jstep + k*kstep],
										&w.x[(map+k*maps)*kernel_size],
										kernel_cols,
										jstep,kernel_cols);
					
					} // k
				} // all maps=chans
			} //g groups
		} 
			
	}


#ifndef MOJO_NO_TRAINING

	// convolution::distribute_delta
	virtual void distribute_delta(base_layer &top, const matrix &w, const int train=1)
	{
		
		// here to calculate top_delta += bottom_delta * W
//		top_delta.x[s] += bottom_delta.x[t]*w.x[s+t*w.cols];
		matrix delta_pad(delta, pad_cols, pad_rows);

		//const int kstep=top.delta.cols*top.delta.rows;
		const int kstep=top.delta.chan_stride;

		const int jstep=top.delta.cols;
		const int output_index=0;
		const int kernel_size=kernel_cols*kernel_rows;
		const int kernel_map_step = kernel_size*kernels_per_map;
		const int map_size=delta_pad.cols*delta_pad.rows;
		const int map_stride=delta_pad.chan_stride;

		const float *_w = w.x;
		const int w_size = kernel_cols;
		const int delta_size = delta_pad.cols;
		const int map_cnt=maps;
		const int top_delta_size = top.delta.rows;
		const int top_delta_chans = top.delta.chans;
		const int stride = _stride;

		matrix delt(top.delta.cols, top.delta.rows, top.delta.chans,NULL,true);



		if (kernel_cols == 5 && stride==1)
		{
			//*
					matrix img_ptr(delta_size, delta_size, 25, NULL, true);
					matrix filter_ptr(28, 1);
	
			const int maps_per_group = map_cnt/groups;
			const int top_chan_per_group = top.delta.chans/groups;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;
					//matrix imgout_ptr(outsize + 7, 1);
					for (int map = start_map; map<stop_map; map++) // how many maps  maps= node.chans
					{
						unwrap_aligned_NxN(5, img_ptr.x, &delta_pad.x[map*map_stride], delta_size, stride);
						const int outsize = top_delta_size*top_delta_size;
						for (int k = start_k; k<stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
						{
							_w = &w.x[(k*maps + map)*kernel_size];
							// flip-flip to make 180 version
							for (int ii = 0; ii < 25; ii++) filter_ptr.x[ii] = _w[24 - ii];
							//float *out = node.x + map_stride*map;
							//float *out = &top.delta.x[k*kstep];
							float *out = &delt.x[k*delt.chan_stride];
							memcpy(out,&top.delta.x[k*kstep],sizeof(float)*outsize);
							dotsum_unwrapped_5x5(img_ptr.x, filter_ptr.x, out, outsize);// imgout_ptr.x, outsize);
							memcpy(&top.delta.x[k*kstep],out,sizeof(float)*outsize);
						}
					}
			}
					/*/
			matrix filter_ptr(28, 1);
			matrix img_ptr(28 * delta_size*delta_size, 1);
			matrix imgout_ptr(delta_size*delta_size, 1);

			for (int map = 0; map < map_cnt; map++) // how many maps  maps= node.chans
			{
				unwrap_aligned_5x5(img_ptr.x, &delta_pad.x[map*map_stride], delta_size, stride);
				const int outsize = top_delta_size*top_delta_size;
				for (int k = 0; k < top_delta_chans; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_w = &w.x[(k*maps + map)*kernel_size];
					// flip-flip to make 180 version
					for (int ii = 0; ii < 25; ii++) filter_ptr.x[ii] = _w[24 - ii];

					dot_unwrapped_5x5(img_ptr.x, filter_ptr.x, imgout_ptr.x, outsize);

					float *out = &top.delta.x[k*kstep];
					for (int j = 0; j < outsize; j++) out[j] += imgout_ptr.x[j];

				}

			}
			//*/
		//	return;
		}
		else if(kernel_cols==3  && stride==1 )
		{
			matrix img_ptr(delta_size, delta_size, 9, NULL, true);
			matrix filter_ptr(9, 1);
					
			const int maps_per_group = map_cnt/groups;
			const int top_chan_per_group = top.delta.chans/groups;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;
				//matrix imgout_ptr(outsize + 7, 1);
				for (int map = start_map; map<stop_map; map++) // how many maps  maps= node.chans
				{
					unwrap_aligned_NxN(3, img_ptr.x, &delta_pad.x[map*map_stride], delta_size, stride);
					const int outsize = top_delta_size*top_delta_size;
					for (int k = start_k; k<stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
					{
						_w = &w.x[(k*maps + map)*kernel_size];
						// flip-flip to make 180 version
						for (int ii = 0; ii < 9; ii++) filter_ptr.x[ii] = _w[8 - ii];
						//float *out = node.x + map_stride*map;
					//	float *out = &top.delta.x[k*kstep];
					//	dotsum_unwrapped_3x3(img_ptr.x, filter_ptr.x, out, outsize);// imgout_ptr.x, outsize);
						float *out = &delt.x[k*delt.chan_stride];
						memcpy(out,&top.delta.x[k*kstep],sizeof(float)*outsize);
						dotsum_unwrapped_3x3(img_ptr.x, filter_ptr.x, out, outsize);// imgout_ptr.x, outsize);
						memcpy(&top.delta.x[k*kstep],out,sizeof(float)*outsize);
					}
				}
			}
		}
		else if (kernel_cols == 2  && stride==1)
		{
			matrix img_ptr(delta_size, delta_size, 4, NULL, true);
			matrix filter_ptr(4, 1);
			matrix out_aligned(top_delta_size,top_delta_size,1,NULL,true);
			const int maps_per_group = map_cnt/groups;
			const int top_chan_per_group = top.delta.chans/groups;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;

			//matrix imgout_ptr(outsize + 7, 1);
			for (int map = start_map; map<stop_map; map++) // how many maps  maps= node.chans
			{
				unwrap_aligned_NxN(2, img_ptr.x, &delta_pad.x[map*map_stride], delta_size, stride);
				const int outsize = top_delta_size*top_delta_size;
				for (int k = start_k; k<stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_w = &w.x[(k*maps + map)*kernel_size];
					// flip-flip to make 180 version
					for (int ii = 0; ii < 4; ii++) filter_ptr.x[ii] = _w[3 - ii];
					memcpy(out_aligned.x, &top.delta.x[k*kstep],outsize*sizeof(float));
					//float *out = node.x + map_stride*map;
					float *out = out_aligned.x;// &top.delta.x[k*kstep];
					dotsum_unwrapped_2x2(img_ptr.x, filter_ptr.x, out, outsize);// imgout_ptr.x, outsize);
					memcpy(&top.delta.x[k*kstep],out_aligned.x,outsize*sizeof(float));

				}
			}
			}
		}
		else if (kernel_cols == 1  && stride==1)
		{
			const int maps_per_group = map_cnt/groups;
			const int top_chan_per_group = top.delta.chans/groups;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;

			for (int j = 0; j<top.delta.rows; j += stride) // input h 
			{
				for (int i = 0; i<top.delta.cols; i += stride) // intput w
				{
					for (int k = start_k; k<stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
					{
						int td_i = i + (j)*jstep + k*kstep;
						float *delt = &delta_pad.x[i + (j)*delta_pad.cols + 0*map_stride];
						float *wx = &w.x[(0 + k*maps)*kernel_size];
						for (int map = start_map; map<stop_map; map++) // how many maps  maps= node.chans
						{
							top.delta.x[td_i] += (*delt)  * (*wx);
							delt += map_stride;
							wx += kernel_size;

						} // all input chans
						  //output_index++;	
					}
				}
			} //y
			}
		}
		else
		{

			const int maps_per_group = map_cnt/groups;
			const int top_chan_per_group = top.delta.chans/groups;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;

				for(int j=0; j<top.delta.rows; j+=stride) // input h 
				{
					for(int i=0; i<top.delta.cols; i+=stride) // intput w
					{

						for(int k=start_k; k<stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
						{
							int td_i = i+(j)*jstep + k*kstep;
							for(int map=start_map; map<stop_map; map++) // how many maps  maps= node.chans
							{
								top.delta.x[td_i] += unwrap_2d_dot_rot180(
									&delta_pad.x[i+(j)*delta_pad.cols + map*map_stride], 
									&w.x[(map+k*maps)*kernel_size],
									kernel_cols,
									delta_pad.cols,kernel_cols);

							} // all input chans
							//output_index++;	
						} 
					}
				} //y
	
			} // groups
		}
	}


	// convolution::calculate_dw
	virtual void calculate_dw(const base_layer &top, matrix &dw, const int train =1)
	{
		int kstep=top.delta.chan_stride;
		int jstep=top.delta.cols;
		int output_index=0;
		int kernel_size=kernel_cols*kernel_rows;
		int kernel_map_step = kernel_size*kernels_per_map;
		int map_size=delta.cols*delta.rows;
		int map_stride=delta.chan_stride;

		dw.resize(kernel_cols, kernel_rows,kernels_per_map*maps);
		dw.fill(0);
		
		// node x already init to 0
		output_index=0;
		const int stride = _stride;
		const int top_node_size= top.node.cols;
		const int node_size = node.rows;
		const int delta_size = delta.cols;
		const int kern_len=kernel_cols;
		const float *_top;
		if(kern_len==5)
		{

				const int maps_per_group = maps/groups;
			const int top_chan_per_group = top.node.chans/groups;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;

			for(int map=start_map; map<stop_map; map++) // how many maps  maps= node.chans
			{
				const float *_delta =&delta.x[map*map_stride];
				for(int k=start_k; k<stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_top = &top.node.x[k*kstep];
					const int w_i = (map+k*maps)*kernel_size;
					const float *_t=_top;
					float *_w=dw.x+w_i;
					_w[0]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[1]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[2]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[3]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[4]+= unwrap_2d_dot( _t, _delta,	node_size,top_node_size, delta_size);
					_t=_top+jstep;
					_w=dw.x+w_i+kern_len;
					_w[0]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[1]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[2]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[3]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[4]+= unwrap_2d_dot( _t, _delta,	node_size,top_node_size, delta_size);
					_t=_top+jstep*2;
					_w=dw.x+w_i+kern_len*2;
					_w[0]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[1]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[2]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[3]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[4]+= unwrap_2d_dot( _t, _delta,	node_size,top_node_size, delta_size);
					_t=_top+jstep*3;
					_w=dw.x+w_i+kern_len*3;
					_w[0]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[1]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[2]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[3]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[4]+= unwrap_2d_dot( _t, _delta,	node_size,top_node_size, delta_size);
					_t=_top+jstep*4;
					_w=dw.x+w_i+kern_len*4;
					_w[0]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[1]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[2]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[3]+= unwrap_2d_dot( _t++, _delta,	node_size,top_node_size, delta_size);
					_w[4]+= unwrap_2d_dot( _t, _delta,	node_size,top_node_size, delta_size);
				} //y
			} // all maps=chans 
			}
		}
		else if(kern_len==3)
		{
			const int maps_per_group = maps/groups;
			const int top_chan_per_group = top.node.chans/groups;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;
			for(int map=start_map; map<stop_map; map++) // how many maps  maps= node.chans
			{
				const float *_delta =&delta.x[map*map_stride];
				for(int k=start_k; k<stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_top = &top.node.x[k*kstep];
					const int w_i = (map+k*maps)*kernel_size;
					dw.x[w_i+0+(0)*kern_len]+= unwrap_2d_dot( _top + 0+(0)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+1+(0)*kern_len]+= unwrap_2d_dot( _top + 1+(0)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+2+(0)*kern_len]+= unwrap_2d_dot( _top + 2+(0)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+0+(1)*kern_len]+= unwrap_2d_dot( _top + 0+(1)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+1+(1)*kern_len]+= unwrap_2d_dot( _top + 1+(1)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+2+(1)*kern_len]+= unwrap_2d_dot( _top + 2+(1)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+0+(2)*kern_len]+= unwrap_2d_dot( _top + 0+(2)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+1+(2)*kern_len]+= unwrap_2d_dot( _top + 1+(2)*jstep, _delta,	node_size,top_node_size, delta_size);
					dw.x[w_i+2+(2)*kern_len]+= unwrap_2d_dot( _top + 2+(2)*jstep, _delta,	node_size,top_node_size, delta_size);
				} //y
			} // all maps=chans 
			}
		}
		else
		{
		
			const int maps_per_group = maps/groups;
			const int top_chan_per_group = top.node.chans/groups;

			for(int g=0; g<groups; g++)
			{	
				const int start_k=0+g*top_chan_per_group;
				const int stop_k=start_k+top_chan_per_group;
				const int start_map=0+g*maps_per_group;
				const int stop_map=start_map+maps_per_group;
			for(int map=start_map; map<stop_map; map++) // how many maps  maps= node.chans
			{
				const float *_delta =&delta.x[map*map_stride];
				for(int k=start_k; k<stop_k; k++) // input channels --- same as kernels_per_map - kern for each input
				{
					_top = &top.node.x[k*kstep];
					const int w_i = (map+k*maps)*kernel_size;
					for(int jj=0; jj<kern_len; jj+=1)
					{
						for(int ii=0; ii<kern_len; ii+=1)
						{
							dw.x[w_i+ii+(jj)*kern_len]+= unwrap_2d_dot( _top + ii+(jj)*jstep, _delta,
								node_size,top_node_size, delta_size);

						} // all input chans
					} // x
				} //y
			} // all maps=chans 
			}
		}
	}

#endif
};


//--------------------------------------------------
// N E W    L A Y E R 
//
// "input", "fully_connected","max_pool","convolution","concatination"
base_layer *new_layer(const char *layer_name, const char *config)
{
	std::istringstream iss(config); 
	std::string str;
	iss>>str;
	int w,h,c,s,g;
	if(str.compare("input")==0)
	{
		iss>>w; iss>>h; iss>>c;
		return new input_layer(layer_name, w,h,c);
	}
	else if(str.compare("fully_connected")==0)
	{
		std::string act;
		iss>>c; iss>>act; 
		if (c<=0) bail("fully_connected layer has invalid output channels");
		//if (act.empty()) bail("fully_connected layer missing activation");
		return new fully_connected_layer(layer_name, c, new_activation_function(act));
	}
	else if (str.compare("softmax") == 0)
	{
		//std::string act;
		iss >> c; //iss >> act;
		if (c<=0) bail("softmax layer has invalid output channels");
		return new fully_connected_layer(layer_name, c, new_activation_function("softmax"));
	}
	/*
	else if (str.compare("brokemax") == 0)
	{
		//std::string act;
		iss >> c; //iss >> act;
		return new fully_connected_layer(layer_name, c, new_activation_function("brokemax"));
	}
	*/
	else if(str.compare("max_pool")==0)
	{
		iss >> c;  iss >> s;
		if(s>0 && s<=c)
			return new max_pooling_layer(layer_name, c, s);
		else
			return new max_pooling_layer(layer_name, c);
	}
	/*
	else if (str.compare("mfm") == 0)
	{
		iss >> c;
		return new maxout_layer(layer_name, c);
	}
	*/
	/*
	else if (str.compare("activation") == 0)
	{
		iss >> s;
		return new activation_layer(layer_name, s);
	}
	*/
	/*
	else if (str.compare("semi_stochastic_pool") == 0)
	{
		iss >> c;  iss >> s;
		if (s>0 && s <= c)
			return new semi_stochastic_pooling_layer(layer_name, c, s);
		else
			return new semi_stochastic_pooling_layer(layer_name, c);
	}
	*/
	/*
	else if (str.compare("deepcnet") == 0)
	{
		std::string act;
		iss >> c; iss >> act;
		return new deepcnet_layer(layer_name, c, new_activation_function(act));
	}
	*/
	else if(str.compare("convolution")==0)
	{
		std::string act;
		iss>>w;iss>>c; iss >> s; iss>>act;
		return new convolution_layer(layer_name, w,c,s, new_activation_function(act));
	}
	else if(str.compare("group_convolution")==0)
	{
		std::string act;
		iss>>w;iss>>c; iss >> s; iss >> g; iss>>act;
		return new convolution_layer(layer_name, w,c,s, g, new_activation_function(act));
	}
	/*
	else if(str.compare("shuffle")==0)
	{
	
		iss >> g; 
		return new shuffle_layer(layer_name, g);
	}
	else if (str.compare("dropout") == 0)
	{
		float fc;
		iss >> fc;
		return new dropout_layer(layer_name, fc);
	}
	else if((str.compare("resize")==0) || (str.compare("concatenate") == 0))
	{
		std::string pad;
		iss>>w; 
		iss >> pad;
		mojo::pad_type p = mojo::zero;
		if (pad.compare("median") == 0) p = mojo::median_edge;
		else if (pad.compare("median_edge") == 0) p = mojo::median_edge;
		else if (pad.compare("edge") == 0) p = mojo::edge;

		return new concatenation_layer(layer_name, w,w, p);
	}	
	*/
	else
	{
		bail("layer type not valid: '" + str + "'");
	}

	return NULL;
}


} // namespace
