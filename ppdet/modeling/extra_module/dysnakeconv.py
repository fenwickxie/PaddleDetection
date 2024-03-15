import paddle
import paddle.nn as nn

__all__ = ['DySnakeConv']


def auto_pad(k, p=None, d=1):  # kernel, padding, dilation
	"""Pad to 'same' shape outputs."""
	if d > 1:
		k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
	if p is None:
		p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
	return p


class Conv(nn.Layer):
	"""Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
	
	default_act = nn.Silu()  # default activation
	
	def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
		"""Initialize Conv layer with given arguments including activation."""
		super().__init__()
		self.conv = nn.Conv2D(c1, c2, k, s, auto_pad(k, p, d), groups=g, dilation=d, bias_attr=False)
		self.bn = nn.BatchNorm2D(c2)
		self.act = self.default_act if act is True else act if isinstance(act, nn.Layer) else nn.Identity()
	
	def forward(self, x):
		"""Apply convolution, batch normalization and activation to input tensor."""
		return self.act(self.bn(self.conv(x)))
	
	def forward_fuse(self, x):
		"""Perform transposed convolution of 2D data."""
		return self.act(self.conv(x))


# Core code, for ease of understanding, we mark the dimensions of input and output next to the code
class DSC(object):
	def __init__(self, input_shape, kernel_size, extend_scope, morph):
		self.num_points = kernel_size
		self.width = input_shape[2]
		self.height = input_shape[3]
		self.morph = morph
		self.extend_scope = extend_scope  # offset (-1 ~ 1) * extend_scope
		
		# define feature map shape
		"""
		B: Batch size  C: Channel  W: Width  H: Height
		"""
		self.num_batch = input_shape[0]
		self.num_channels = input_shape[1]
	
	"""
	input: offset [B,2*K,W,H]  K: Kernel size (2*K: 2D image, deformation contains <x_offset> and <y_offset>)
	output_x: [B,1,W,K*H]   coordinate map
	output_y: [B,1,K*W,H]   coordinate map
	"""
	
	def _coordinate_map_3D(self, offset, if_offset):
		place = offset.place
		# offset
		y_offset, x_offset = paddle.split(offset, 2, axis=1)
		
		y_center = paddle.arange(0, self.width).tile([self.height])
		y_center = y_center.reshape([self.height, self.width])
		y_center = y_center.transpose([1, 0])
		y_center = y_center.reshape([-1, self.width, self.height])
		y_center = y_center.tile([self.num_points, 1, 1]).astype('float32')
		y_center = y_center.unsqueeze(0)
		
		x_center = paddle.arange(0, self.height).tile([self.width])
		x_center = x_center.reshape([self.width, self.height])
		x_center = x_center.transpose([0, 1])
		x_center = x_center.reshape([-1, self.width, self.height])
		x_center = x_center.tile([self.num_points, 1, 1]).astype('float32')
		x_center = x_center.unsqueeze(0)
		
		if self.morph == 0:
			"""
			Initialize the kernel and flatten the kernel
				y: only need 0
				x: -num_points//2 ~ num_points//2 (Determined by the kernel size)
				!!! The related PPT will be submitted later, and the PPT will contain the whole changes of each step
			"""
			y = paddle.linspace(0, 0, 1)
			x = paddle.linspace(
					-int(self.num_points // 2),
					int(self.num_points // 2),
					int(self.num_points),
			)
			
			y, x = paddle.meshgrid(y, x, indexing='ij')
			y_spread = y.reshape([-1, 1])
			x_spread = x.reshape([-1, 1])
			
			y_grid = y_spread.tile([1, self.width * self.height])
			y_grid = y_grid.reshape([self.num_points, self.width, self.height])
			y_grid = y_grid.unsqueeze(0)  # [B*K*K, W,H]
			
			x_grid = x_spread.tile([1, self.width * self.height])
			x_grid = x_grid.reshape([self.num_points, self.width, self.height])
			x_grid = x_grid.unsqueeze(0)  # [B*K*K, W,H]
			
			y_new = y_center + y_grid
			x_new = x_center + x_grid
			
			y_new = paddle.to_tensor(y_new.tile([self.num_batch, 1, 1, 1]), place=place)
			x_new = paddle.to_tensor(x_new.tile([self.num_batch, 1, 1, 1]), place=place)
			
			y_offset_new = y_offset.detach().clone()
			
			if if_offset:
				y_offset = y_offset.transpose([1, 0, 2, 3])
				y_offset_new = y_offset_new.transpose([1, 0, 2, 3])
				center = int(self.num_points // 2)
				
				# The center position remains unchanged and the rest of the positions begin to swing
				# This part is quite simple. The main idea is that "offset is an iterative process"
				y_offset_new[center] = 0
				for index in range(1, center):
					y_offset_new[center + index] = (y_offset_new[center + index - 1] + y_offset[center + index])
					y_offset_new[center - index] = (y_offset_new[center - index + 1] + y_offset[center - index])
				y_offset_new = paddle.to_tensor(y_offset_new.transpose([1, 0, 2, 3]), place=place)
				y_new = y_new.add(y_offset_new.multiply(y=paddle.to_tensor(self.extend_scope, dtype='float32', place=place)))
			
			y_new = y_new.reshape(
					[self.num_batch, self.num_points, 1, self.width, self.height])
			y_new = y_new.transpose([0, 3, 1, 4, 2])
			y_new = y_new.reshape([
				self.num_batch, self.num_points * self.width, 1 * self.height
			])
			x_new = x_new.reshape(
					[self.num_batch, self.num_points, 1, self.width, self.height])
			x_new = x_new.transpose([0, 3, 1, 4, 2])
			x_new = x_new.reshape([
				self.num_batch, self.num_points * self.width, 1 * self.height
			])
			return y_new, x_new
		
		else:
			"""
			Initialize the kernel and flatten the kernel
				y: -num_points//2 ~ num_points//2 (Determined by the kernel size)
				x: only need 0
			"""
			y = paddle.linspace(
					-int(self.num_points // 2),
					int(self.num_points // 2),
					int(self.num_points),
			)
			x = paddle.linspace(0, 0, 1)
			
			y, x = paddle.meshgrid(y, x, indexing='ij')
			y_spread = y.reshape([-1, 1])
			x_spread = x.reshape([-1, 1])
			
			y_grid = y_spread.tile([1, self.width * self.height])
			y_grid = y_grid.reshape([self.num_points, self.width, self.height])
			y_grid = y_grid.unsqueeze(0)
			
			x_grid = x_spread.tile([1, self.width * self.height])
			x_grid = x_grid.reshape([self.num_points, self.width, self.height])
			x_grid = x_grid.unsqueeze(0)
			
			y_new = y_center + y_grid
			x_new = x_center + x_grid
			
			y_new = y_new.tile([self.num_batch, 1, 1, 1])
			x_new = x_new.tile([self.num_batch, 1, 1, 1])
			
			y_new = paddle.to_tensor(y_new, place=place)
			x_new = paddle.to_tensor(x_new, place=place)
			x_offset_new = x_offset.detach().clone()
			
			if if_offset:
				x_offset = x_offset.transpose([1, 0, 2, 3])
				x_offset_new = x_offset_new.transpose([1, 0, 2, 3])
				center = int(self.num_points // 2)
				x_offset_new[center] = 0
				for index in range(1, center):
					x_offset_new[center + index] = (x_offset_new[center + index - 1] + x_offset[center + index])
					x_offset_new[center - index] = (x_offset_new[center - index + 1] + x_offset[center - index])
				x_offset_new = paddle.to_tensor(x_offset_new.transpose([1, 0, 2, 3]), place=place)
				x_new = x_new.add(x_offset_new.multiply(y=paddle.to_tensor(self.extend_scope, dtype='float32', place=place)))
			
			y_new = y_new.reshape(
					[self.num_batch, 1, self.num_points, self.width, self.height])
			y_new = y_new.transpose([0, 3, 1, 4, 2])
			y_new = y_new.reshape([
				self.num_batch, 1 * self.width, self.num_points * self.height
			])
			x_new = x_new.reshape(
					[self.num_batch, 1, self.num_points, self.width, self.height])
			x_new = x_new.transpose([0, 3, 1, 4, 2])
			x_new = x_new.reshape([
				self.num_batch, 1 * self.width, self.num_points * self.height
			])
			return y_new, x_new
	
	"""
	input: input feature map [N,C,D,W,H]；coordinate map [N,K*D,K*W,K*H]
	output: [N,1,K*D,K*W,K*H]  deformed feature map
	"""
	
	def _bilinear_interpolate_3D(self, input_feature, y, x):
		place = input_feature.place
		y = y.reshape([-1]).astype('float32')
		x = x.reshape([-1]).astype('float32')
		
		zero = paddle.zeros([]).astype('int64')
		max_y = self.width - 1
		max_x = self.height - 1
		
		# find 8 grid locations
		y0 = paddle.floor(y).astype('int64')
		y1 = y0 + 1
		x0 = paddle.floor(x).astype('int64')
		x1 = x0 + 1
		
		# clip out coordinates exceeding feature map volume
		y0 = paddle.clip(y0, zero, max_y)
		y1 = paddle.clip(y1, zero, max_y)
		x0 = paddle.clip(x0, zero, max_x)
		x1 = paddle.clip(x1, zero, max_x)
		
		input_feature_flat = input_feature.flatten()
		input_feature_flat = input_feature_flat.reshape([self.num_batch, self.num_channels, self.width, self.height])
		input_feature_flat = input_feature_flat.transpose([0, 2, 3, 1])
		input_feature_flat = input_feature_flat.reshape([-1, self.num_channels])
		dimension = self.height * self.width
		
		base = paddle.arange(self.num_batch) * dimension
		base = base.reshape([-1, 1]).astype('float32')
		
		repeat = paddle.ones([self.num_points * self.width * self.height
							  ]).unsqueeze(0)
		repeat = repeat.astype('float32')
		
		base = paddle.matmul(base, repeat)
		base = base.reshape([-1])
		
		base = paddle.to_tensor(base, place=place)
		
		base_y0 = base + y0 * self.height
		base_y1 = base + y1 * self.height
		
		# top rectangle of the neighbourhood volume
		index_a0 = base_y0 - base + x0
		index_c0 = base_y0 - base + x1
		
		# bottom rectangle of the neighbourhood volume
		index_a1 = base_y1 - base + x0
		index_c1 = base_y1 - base + x1
		
		# get 8 grid values
		value_a0 = paddle.to_tensor(input_feature_flat[index_a0.astype('int64')], place=place)
		value_c0 = paddle.to_tensor(input_feature_flat[index_c0.astype('int64')], place=place)
		value_a1 = paddle.to_tensor(input_feature_flat[index_a1.astype('int64')], place=place)
		value_c1 = paddle.to_tensor(input_feature_flat[index_c1.astype('int64')], place=place)
		
		# find 8 grid locations
		y0 = paddle.floor(y).astype('int64')
		y1 = y0 + 1
		x0 = paddle.floor(x).astype('int64')
		x1 = x0 + 1
		
		# clip out coordinates exceeding feature map volume
		y0 = paddle.clip(y0, zero, max_y + 1)
		y1 = paddle.clip(y1, zero, max_y + 1)
		x0 = paddle.clip(x0, zero, max_x + 1)
		x1 = paddle.clip(x1, zero, max_x + 1)
		
		x0_float = x0.astype('float32')
		x1_float = x1.astype('float32')
		y0_float = y0.astype('float32')
		y1_float = y1.astype('float32')
		
		vol_a0 = paddle.to_tensor(((y1_float - y) * (x1_float - x)).unsqueeze(-1), place=place)
		vol_c0 = paddle.to_tensor(((y1_float - y) * (x - x0_float)).unsqueeze(-1), place=place)
		vol_a1 = paddle.to_tensor(((y - y0_float) * (x1_float - x)).unsqueeze(-1), place=place)
		vol_c1 = paddle.to_tensor(((y - y0_float) * (x - x0_float)).unsqueeze(-1), place=place)
		
		outputs = (value_a0 * vol_a0 + value_c0 * vol_c0 + value_a1 * vol_a1 +
				   value_c1 * vol_c1)
		
		if self.morph == 0:
			outputs = outputs.reshape([
				self.num_batch,
				self.num_points * self.width,
				1 * self.height,
				self.num_channels,
				])
			outputs = outputs.transpose([0, 3, 1, 2])
		else:
			outputs = outputs.reshape([
				self.num_batch,
				1 * self.width,
				self.num_points * self.height,
				self.num_channels,
				])
			outputs = outputs.transpose([0, 3, 1, 2])
		return outputs
	
	def deform_conv(self, input, offset, if_offset):
		y, x = self._coordinate_map_3D(offset, if_offset)
		deformed_feature = self._bilinear_interpolate_3D(input, y, x)
		return deformed_feature


class DSConv(nn.Layer):
	def __init__(self, in_ch, out_ch, morph, kernel_size=3, if_offset=True, extend_scope=1):
		"""
		The Dynamic Snake Convolution
		:param in_ch: input channel
		:param out_ch: output channel
		:param kernel_size: the size of kernel
		:param extend_scope: the range to expand (default 1 for this method)
		:param morph: the morphology of the convolution kernel is mainly divided into two types
						along the x-axis (0) and the y-axis (1) (see the paper for details)
		:param if_offset: whether deformation is required, if it is False, it is the standard convolution kernel
		"""
		super(DSConv, self).__init__()
		# use the <offset_conv> to learn the deformable offset
		self.offset_conv = nn.Conv2D(in_ch, 2 * kernel_size, 3, padding=1)
		self.bn = nn.BatchNorm2D(2 * kernel_size)
		self.kernel_size = kernel_size
		
		# two types of the DSConv (along x-axis and y-axis)
		self.dsc_conv_x = nn.Conv2D(
				in_ch,
				out_ch,
				kernel_size=(kernel_size, 1),
				stride=(kernel_size, 1),
				padding=0,
		)
		self.dsc_conv_y = nn.Conv2D(
				in_ch,
				out_ch,
				kernel_size=(1, kernel_size),
				stride=(1, kernel_size),
				padding=0,
		)
		
		self.gn = nn.GroupNorm(out_ch // 4, out_ch)
		self.act = Conv.default_act
		
		self.extend_scope = extend_scope
		self.morph = morph
		self.if_offset = if_offset
	
	def forward(self, f):
		offset = self.offset_conv(f)
		offset = self.bn(offset)
		# We need a range of deformation between -1 and 1 to mimic the snake's swing
		offset = paddle.tanh(offset)
		input_shape = f.shape
		dsc = DSC(input_shape, self.kernel_size, self.extend_scope, self.morph)
		deformed_feature = dsc.deform_conv(f, offset, self.if_offset)
		if self.morph == 0:
			x = self.dsc_conv_x(deformed_feature.astype(f.dtype))
			x = self.gn(x)
			x = self.act(x)
			return x
		else:
			x = self.dsc_conv_y(deformed_feature.astype(f.dtype))
			x = self.gn(x)
			x = self.act(x)
			return x


class DySnakeConv(nn.Layer):
	def __init__(self, inc, ouc, k=3) -> None:
		super().__init__()
		c_ = ouc // 3 // 16 * 16
		self.conv_0 = Conv(inc, ouc - 2 * c_, k)
		self.conv_x = DSConv(inc, c_, 0, k)
		self.conv_y = DSConv(inc, c_, 1, k)
	
	def forward(self, x):
		return paddle.concat([self.conv_0(x), self.conv_x(x), self.conv_y(x)], axis=1)
