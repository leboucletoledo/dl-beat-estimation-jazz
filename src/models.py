from torch import (nn as nn, 
                   cat as t_cat, 
                   add as t_add, 
                   squeeze as t_squeeze,
                   ne as t_ne,
                   HalfTensor,
                   FloatTensor,
                   )

class ResidualBlock(nn.Module):
    def __init__(self, 
                 n_inputs, 
                 n_filters, 
                 k_size,
                 padding, 
                 dilation, 
                 do_rate=0.2,
                 ):
        super(ResidualBlock, self).__init__()
        self.n_inputs = n_inputs
        self.n_filters = n_filters
        self.k_size = k_size
        self.padding = padding
        self.dilation = dilation
        self.do_rate = do_rate
        # Create residual from input
        self.res_x = nn.Conv1d(in_channels=self.n_inputs,
                                     out_channels=self.n_filters,
                                     kernel_size=1,
                                     padding='same',
                                     )
        # Create concatenated convolutional layers with different dilation
        self.conv1d_dil_1 = nn.Conv1d(in_channels=self.n_inputs,
                                            out_channels=self.n_filters,
                                            kernel_size=self.k_size,
                                            padding=self.padding,
                                            dilation=self.dilation,
                                            )
        self.conv1d_dil_2 = nn.Conv1d(in_channels=self.n_inputs,
                                            out_channels=self.n_filters,
                                            kernel_size=self.k_size,
                                            padding=self.padding,
                                            dilation=self.dilation * 2,
                                            )
        # ELU activation
        self.elu = nn.ELU()
        # Downstream 1-D Convolution 
        # Input must match output from parallel dilated convs
        self.conv1d_downstream = nn.Conv1d(in_channels=n_filters * 2,
                                                 out_channels=n_filters,
                                                 kernel_size=1,
                                                 padding='same',
                                                 )
    
    def _concatenate_conv(self, x):
        x = t_cat((self.conv1d_dil_1(x), self.conv1d_dil_2(x)), dim=1) 
        return x
        
    def _spatial_dropout1d(self, x):
        '''Emulate SpatialDropout1D from TF using Dropout1d'''
        do1d = nn.Dropout1d(p=self.do_rate)
        # convert to [batch, channels, time]
        x = do1d(x.permute(0, 2, 1))
        # back to [batch, time, channels] 
        x = x.permute(0, 2, 1)       
        return x

    def forward(self, x):
        assert x.shape[1] == self.n_inputs, "Input channel parameter does not match tensor shape"
        res = self.res_x(x)
        x = self._concatenate_conv(x)
        # Pre-Spatial Dropout
        x = self.elu(x) 
        # Post-Spatial Dropout
        x = self._spatial_dropout1d(x) 
        x = self.conv1d_downstream(x)
        merge = t_add(x, res)

        # For skip connections, use 'return merge, x' 
        return merge 
        
class TCN(nn.Module):
    def __init__(self, 
                 num_inputs, 
                 num_filters=20,
                 kernel_size=5,
                 num_layers=11,
                 padding='same', 
                 dropout_rate=0.1
                 ):
        super(TCN, self).__init__()
        self.num_inputs = num_inputs
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.padding = padding
        assert padding=='same', "Padding type not supported, only 'same'." 
        self.dropout_rate = dropout_rate
        # Create residual blocks        
        self.res_blocks = self._make_blocks()
        # Activation function for TCN stack
        self.elu = nn.ELU()

    def _make_blocks(self):
        blocks = []
        for i in range(self.num_layers):
            dilation = 2 ** i
            n_inputs = self.num_inputs if i==0 else self.num_filters
            blocks.append(ResidualBlock(n_inputs=n_inputs,
                                        n_filters=self.num_filters,
                                        k_size=self.kernel_size,
                                        padding=self.padding,
                                        dilation=dilation,
                                        do_rate=self.dropout_rate,
                                        )
                          )
        return nn.Sequential(*blocks)
    
    def forward(self, x):
        assert (x.shape[1] == self.num_inputs), "Input channel parameter does not match tensor shape"
        x = self.res_blocks(x)
        x = self.elu(x)
        return x

class BeatTrackerModel(nn.Module):
    def __init__(self, num_channels=1, 
                 num_filters=20, 
                 kernel_size=5, 
                 num_layers=11, 
                 padding='same', 
                 dropout_rate=0.2,
                 ):
        super(BeatTrackerModel, self).__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.kernel_size= kernel_size
        self.num_layers = num_layers
        self.padding = padding
        self.dropout_rate = dropout_rate    
        # Create stack of convolutions
        self.conv_stack = self._make_stack(self.num_channels, 
                                           self.num_filters, 
                                           self.dropout_rate,
                                           )
        # Build TCN layers
        self.TCN = TCN(num_inputs=self.num_filters, 
                       num_filters=self.num_filters, 
                       kernel_size=self.kernel_size, 
                       num_layers=self.num_layers, 
                       padding=self.padding, 
                       dropout_rate=self.dropout_rate,
                       )
        # Create output for beats
        self.beat_out = nn.Sequential(nn.Dropout(p=dropout_rate),
                                            nn.Linear(self.num_filters, 1),
                                            nn.Sigmoid(),
                                            )
        # Create output for downbeats
        self.downbeat_out = nn.Sequential(nn.Dropout(p=dropout_rate),
                                                nn.Linear(self.num_filters, 1),
                                                nn.Sigmoid(),
                                                ) 
            
    def _make_layer(self, in_ch, out_ch, ker, drop):
        return nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=ker, padding='valid'),
                                   nn.ELU(),
                                   nn.MaxPool2d((1, 3)),
                                   nn.Dropout(p=drop))
    
    def _make_stack(self, in_ch, out_ch, drop):
        """Build a stack of convolutional layers"""
        layers = []
        kernel_stack = [(3, 3), (1, 10), (3, 3)]
        layers = [
                  self._make_layer(in_ch, out_ch, kernel_stack[i], drop) if i==0 
                  else self._make_layer(out_ch, out_ch, kernel_stack[i], drop)
                  for i in range(0, 3)
                  ]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv_stack(x)
        x = t_squeeze(x, dim=-1)
        x = self.TCN(x)
        x = x.permute(0, 2, 1)
        beats = self.beat_out(x)
        downbeats = self.downbeat_out(x)
        return beats.permute(0, 2, 1), downbeats.permute(0, 2, 1)
    
class LossFunctionMasked(nn.Module):
    def __init__(self, 
                 loss_function, 
                 mask=None,
                 device=None,
                 half_tensor=False,
                 ):
        super(LossFunctionMasked, self).__init__()
        self.loss_fun = loss_function
        self.mask = mask
        self.device = device
        # Use Half-Tensor might alleviate resource usage, use with caution
        self.ht = half_tensor
    
    # TODO: add device as argument to loss call instances in main 
    def forward(self, 
                outputs, 
                targets,
                ):        
        if self.mask is None:
            return self.loss_fun(outputs, targets,)
        else:
            mask_arr = t_ne(targets, self.mask)
            if self.ht: 
                mask_arr = mask_arr.type(dtype=HalfTensor).to(self.device)
            else: 
                mask_arr = mask_arr.type(dtype=FloatTensor).to(self.device)
            return self.loss_fun(outputs * mask_arr, targets * mask_arr,)