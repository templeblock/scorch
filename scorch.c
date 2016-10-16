
#include "scorch.h"

#define INITIAL_CONTAINER_SIZE 20

SCH_Network *SCH_SequentialNetwork()
{
  SCH_Network *net = malloc(sizeof(SCH_Network));
  net->layers = malloc(INITIAL_CONTAINER_SIZE*sizeof(SCH_Layer*));
  net->loss = NULL;
  net->end = 0;
  net->size = INITIAL_CONTAINER_SIZE;
  return net;
}

void SCH_SequentialAdd(SCH_Network *net, SCH_Layer *layer)
{
  if (net->end == net->size-1) {
    SCH_Layer **layers = malloc(2*net->size*sizeof(SCH_Layer*));
    memcpy(layers,net->layers,net->end*sizeof(SCH_Layer*));
    free(net->layers);
    net->layers = layers;
  }
  net->layers[net->end] = layer;
  net->end += 1;
}

void SCH_SequentialAddLoss(SCH_Network *net, SCH_Loss *loss)
{
  net->loss = loss;
}

void SCH_LayerDelete(SCH_Layer *layer)
{
  THFloatTensor_free(layer->output);
  THFloatTensor_free(layer->gradInput);
  layer->freeState(layer);
  free(layer);
}

void SCH_LossDelete(SCH_Loss *loss)
{
  THFloatTensor_free(loss->output);
  THFloatTensor_free(loss->gradInput);
  loss->freeState(loss);
  free(loss);
}

void SCH_NetworkDelete(SCH_Network *net)
{
  for (long i=0; i<net->end; i++) {
    SCH_LayerDelete(net->layers[i]);
  }
  if (net->loss) {
    SCH_LossDelete(net->loss);
  }
  free(net);
}

THFloatTensor *SCH_NetworkForward(SCH_Network *net,
                                  THFloatTensor *input)
{
  THFloatTensor *output = input;
  for (int i=net->end-1; i>=0; i--) {
    output = SCH_LayerForward(net->layers[i],output);
  }
  return output;
}

THFloatTensor *SCH_NetworkBackward(SCH_Network *net,
                                   THFloatTensor *input,
                                   THFloatTensor *gradOutput,
                                   float scale)
{
  THFloatTensor *gradInput = gradOutput;
  for (int i=net->end-1; i>=0; i--) {
    THFloatTensor *layerInput = i>0 ? net->layers[i-1]->output : input;
    gradInput = SCH_LayerBackward(net->layers[i],
                                  layerInput,
                                  gradInput,
                                  scale);
  }
  return gradInput;
}

THFloatTensor *SCH_NetworkForwardWithLoss(SCH_Network *net,
                                          THFloatTensor *input,
                                          THFloatTensor *target)
{
  THFloatTensor *output = SCH_NetworkForward(net,input);
  return SCH_LossForward(net->loss,output,target);
}

THFloatTensor *SCH_NetworkBackwardWithLoss(SCH_Network *net,
                                           THFloatTensor *input,
                                           THFloatTensor *target,
                                           float scale)
{
  THFloatTensor *gradOutput = SCH_LossBackward(net->loss,
                                               net->layers[net->end-1]->output,
                                               target);
  return SCH_NetworkBackward(net,input,gradOutput,scale);
}


THFloatTensor *SCH_LayerForward(SCH_Layer *layer,
                                THFloatTensor *input)
{
  layer->updateOutput(layer,input);
  return layer->output;
}

THFloatTensor *SCH_LayerBackward(SCH_Layer *layer,
                                 THFloatTensor *input,
                                 THFloatTensor *gradOutput,
                                 float scale)
{
  layer->updateGradInput(layer,input,gradOutput);
  if (layer->accGradParameters) {
    layer->accGradParameters(layer,input,gradOutput,scale);
  }
  return layer->gradInput;
}

THFloatTensor *SCH_LossForward(SCH_Loss *loss,
                               THFloatTensor *input,
                               THFloatTensor *target)
{
  loss->updateOutput(loss,input,target);
  return loss->output;
}

THFloatTensor *SCH_LossBackward(SCH_Loss *loss,
                                THFloatTensor *input,
                                THFloatTensor *target)
{
  loss->updateGradInput(loss,input,target);
  return loss->gradInput;
}



typedef struct SCH_ELUState {
  float alpha;
  bool inplace;
} SCH_ELUState;

void SCH_ELU_freeState(SCH_Layer *layer)
{
  SCH_ELUState *state = (SCH_ELUState*)layer->state;
  free(state);
}

void SCH_ELU_updateOutput(SCH_Layer *layer,
                          THFloatTensor *input)
{
  SCH_ELUState *state = (SCH_ELUState*)layer->state;
  THNN_FloatELU_updateOutput(NULL,
                             input,
                             layer->output,
                             state->alpha,
                             state->inplace);
}

void SCH_ELU_updateGradInput(SCH_Layer *layer,
                             THFloatTensor *input,
                             THFloatTensor *gradOutput)
{
  SCH_ELUState *state = (SCH_ELUState*)layer->state;
  THNN_FloatELU_updateGradInput(NULL,
                                input,
                                gradOutput,
                                layer->gradInput,
                                layer->output,
                                state->alpha,
                                state->inplace);
}

SCH_Layer *SCH_ELULayer(float alpha, bool inplace)
{
  SCH_ELUState *state = malloc(sizeof(SCH_ELUState));
  state->alpha = alpha;
  state->inplace = inplace;

  SCH_Layer *layer = malloc(sizeof(SCH_Layer));
  layer->updateOutput = SCH_ELU_updateOutput;
  layer->updateGradInput = SCH_ELU_updateGradInput;
  layer->accGradParameters = NULL;
  layer->gradInput = THFloatTensor_new();
  layer->output = THFloatTensor_new();
  layer->state = state;
  layer->freeState = SCH_ELU_freeState;

  return layer;
}



typedef struct SCH_LinearState {
  THFloatTensor *weight;
  THFloatTensor *gradWeight;
  THFloatTensor *bias;
  THFloatTensor *gradBias;
  THFloatTensor *addBuffer;
} SCH_LinearState;

// TODO reset()

void SCH_Linear_freeState(SCH_Layer *layer)
{
  SCH_LinearState *state = (SCH_LinearState*)layer->state;
  THFloatTensor_free(state->weight);
  THFloatTensor_free(state->gradWeight);
  THFloatTensor_free(state->bias);
  THFloatTensor_free(state->gradBias);
  THFloatTensor_free(state->addBuffer);
  free(state);
}

void SCH_Linear_updateOutput(SCH_Layer *layer,
                             THFloatTensor *input)
{
  SCH_LinearState *state = (SCH_LinearState*)layer->state;
  THNN_FloatLinear_updateOutput(NULL,
                                input,
                                layer->output,
                                state->weight,
                                state->bias,
                                state->addBuffer);
}

void SCH_Linear_updateGradInput(SCH_Layer *layer,
                                THFloatTensor *input,
                                THFloatTensor *gradOutput)
{
  SCH_LinearState *state = (SCH_LinearState*)layer->state;
  THNN_FloatLinear_updateGradInput(NULL,
                                   input,
                                   gradOutput,
                                   layer->gradInput,
                                   state->weight);
}

void SCH_Linear_accGradParameters(SCH_Layer *layer,
                                  THFloatTensor *input,
                                  THFloatTensor *gradOutput,
                                  float scale)
{
  SCH_LinearState *state = (SCH_LinearState*)layer->state;
  THNN_FloatLinear_accGradParameters(NULL,
                                     input,
                                     gradOutput,
                                     layer->gradInput,
                                     state->weight,
                                     state->bias,
                                     state->gradWeight,
                                     state->gradBias,
                                     state->addBuffer,
                                     scale);
}

SCH_Layer *SCH_LinearLayer(long inputSize,
                           long outputSize,
                           bool bias)
{
  SCH_LinearState *state = malloc(sizeof(SCH_LinearState));
  state->weight = THFloatTensor_newWithSize2d(outputSize,inputSize);
  state->gradWeight = THFloatTensor_newWithSize2d(outputSize,inputSize);
  state->addBuffer = THFloatTensor_newWithSize1d(outputSize);
  if (bias) {
    state->bias = THFloatTensor_newWithSize1d(outputSize);
    state->gradBias = THFloatTensor_newWithSize1d(outputSize);
  }
  else {
    state->bias = NULL;
    state->gradBias = NULL;
  }

  SCH_Layer *layer = malloc(sizeof(SCH_Layer));
  layer->updateOutput = SCH_Linear_updateOutput;
  layer->updateGradInput = SCH_Linear_updateGradInput;
  layer->accGradParameters = SCH_Linear_accGradParameters;
  layer->gradInput = THFloatTensor_new();
  layer->output = THFloatTensor_new();
  layer->state = state;
  layer->freeState = SCH_Linear_freeState;

  return layer;
}


typedef struct SCH_MSELossState {
  float sizeAverage;
} SCH_MSELossState;

void SCH_MSELoss_freeState(SCH_Loss *loss)
{
  SCH_MSELossState *state = (SCH_MSELossState*)loss->state;
  free(state);
}

void SCH_MSELoss_updateOutput(SCH_Loss *loss,
                              THFloatTensor *input,
                              THFloatTensor *target)
{
  SCH_MSELossState *state = (SCH_MSELossState*)loss->state;
  if (loss->output->size == NULL) {
    THFloatTensor_resize1d(loss->output,1);
  }
  THNN_FloatMSECriterion_updateOutput(NULL,
                                      input,
                                      target,
                                      loss->output,
                                      state->sizeAverage);
}

void SCH_MSELoss_updateGradInput(SCH_Loss *loss,
                                 THFloatTensor *input,
                                 THFloatTensor *target)
{
  SCH_MSELossState *state = (SCH_MSELossState*)loss->state;
  THNN_FloatMSECriterion_updateGradInput(NULL,
                                         input,
                                         target,
                                         loss->gradInput,
                                         state->sizeAverage);
}

SCH_Loss *SCH_MSELoss(float sizeAverage)
{
  SCH_MSELossState *state = malloc(sizeof(SCH_MSELossState));
  state->sizeAverage = sizeAverage;

  SCH_Loss *loss = malloc(sizeof(SCH_Loss));
  loss->updateOutput = SCH_MSELoss_updateOutput;
  loss->updateGradInput = SCH_MSELoss_updateGradInput;
  loss->gradInput = THFloatTensor_new();
  loss->output = THFloatTensor_new();
  loss->state = state;
  loss->freeState = SCH_MSELoss_freeState;

  return loss;
}


int main(int argc, char *argv[]) {

  SCH_Network *net = SCH_SequentialNetwork();

  SCH_SequentialAdd(net,SCH_ELULayer(1.0,true));
  SCH_SequentialAdd(net,SCH_LinearLayer(10,10,true));
  SCH_SequentialAddLoss(net,SCH_MSELoss(1.0));

  THFloatTensor *input = THFloatTensor_newWithSize2d(10,10);
  THFloatTensor *gradOutput = THFloatTensor_newWithSize2d(10,10);
  THFloatTensor *target = THFloatTensor_newWithSize2d(10,10);

  THFloatTensor *output = SCH_NetworkForward(net,input);
  THFloatTensor *gradInput = SCH_NetworkBackward(net,input,gradOutput,1.0);

  THFloatTensor *loss = SCH_NetworkForwardWithLoss(net,input,target);
  THFloatTensor *gradInputOfLoss = SCH_NetworkBackwardWithLoss(net,input,target,1.0);

  printf("%s\n",THFloatTensor_desc(output).str);
  printf("%s\n",THFloatTensor_desc(gradInput).str);

  printf("%s\n",THFloatTensor_desc(loss).str);
  printf("%s\n",THFloatTensor_desc(gradInputOfLoss).str);

  SCH_NetworkDelete(net);

  return 0;
}

