#ifndef __CORSCH_H__
#define __CORSCH_H__

#include "TH.h"
#include "THNN.h"

typedef struct SCH_Layer {
  void (*updateOutput)(struct SCH_Layer *self,
                       THFloatTensor *input);
  void (*updateGradInput)(struct SCH_Layer *self,
                          THFloatTensor *input,
                          THFloatTensor *gradOutput);
  void (*accGradParameters)(struct SCH_Layer *self,
                            THFloatTensor *input,
                            THFloatTensor *gradOutput,
                            float scale);
  void (*freeState)(struct SCH_Layer *self);
  THFloatTensor *output;
  THFloatTensor *gradInput;
  void *state;
} SCH_Layer;

typedef struct SCH_Loss {
  void (*updateOutput)(struct SCH_Loss *self,
                       THFloatTensor *input,
                       THFloatTensor *target);
  void (*updateGradInput)(struct SCH_Loss *self,
                          THFloatTensor *input,
                          THFloatTensor *target);
  void (*freeState)(struct SCH_Loss *self);
  THFloatTensor *output;
  THFloatTensor *gradInput;
  void *state;
} SCH_Loss;

typedef struct SCH_Network {
  SCH_Layer **layers;
  SCH_Loss *loss; // TODO: extend to more than one loss
  long end;
  long size;
} SCH_Network;

SCH_Network *SCH_SequentialNetwork();

//SCH_Network *SCH_NetworkClone(SCH_Network *net);

void SCH_SequentialAdd(SCH_Network *net, SCH_Layer *layer);

void SCH_SequentialAddLoss(SCH_Network *net, SCH_Loss *loss);

void SCH_LayerDelete(SCH_Layer *layer);

void SCH_NetworkDelete(SCH_Network *net);

THFloatTensor *SCH_NetworkForward(SCH_Network *net,
                                 THFloatTensor *input);

THFloatTensor *SCH_NetworkBackward(SCH_Network *net,
                                  THFloatTensor *input,
                                  THFloatTensor *gradOutput,
                                  float scale);

THFloatTensor *SCH_NetworkForwardWithLoss(SCH_Network *net,
                                         THFloatTensor *input,
                                         THFloatTensor *target);

THFloatTensor *SCH_NetworkBackwardWithLoss(SCH_Network *net,
                                          THFloatTensor *input,
                                          THFloatTensor *target,
                                          float scale);

THFloatTensor *SCH_LayerForward(SCH_Layer *layer,
                               THFloatTensor *input);

THFloatTensor *SCH_LayerBackward(SCH_Layer *layer,
                                THFloatTensor *input,
                                THFloatTensor *gradOutput,
                                float scale);

THFloatTensor *SCH_LossForward(SCH_Loss *loss,
                              THFloatTensor *input,
                              THFloatTensor *target);

THFloatTensor *SCH_LossBackward(SCH_Loss *loss,
                               THFloatTensor *input,
                               THFloatTensor *target);

SCH_Layer *SCH_ELULayer(float alpha,
                      bool inplace);

SCH_Layer *SCH_LinearLayer(long inputSize,
                         long outputSize,
                         bool bias);

SCH_Loss *SCH_MSELoss(float sizeAverage);



/*
void SCH_Train(SCH_Network *net,
              SCH_Optimizer *opt,
              float *inputs,
              long *outputs,
              long n_samples,
              long sample_size,
              long batch_size,
              long epochs,
              void (*batch_cb)(SCH_Network *net, void *data),
              void (*epoch_cb)(SCH_Network *net, void *data),
              void *cb_data,
              long reset_weights, // false
              long n_threads);

void SCH_Predict(SCH_Network *net,
                float *input,
                float *output,
                long input_size,
                long output_size);

long SCH_PredictLabel(SCH_Network *net,
                     float *input,
                     long input_size);

float SCH_GetLoss(SCH_Network *net,
                 float *inputs,
                 float *outputs,
                 long n_samples,
                 long sample_size,
                 long output_size);
*/

#endif

