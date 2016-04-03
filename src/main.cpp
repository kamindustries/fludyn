/*//////////////////////////////////////////////////////////////////////////////
	Kurt Kaminski
	ARG
	2016
*///////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include "allocore/al_Allocore.hpp"

#include "kernels.cuh"
#include "cudaFunc.cuh"

using namespace al;
using namespace std;

// #define MAX(a,b) ((a > b) ? a : b)
// #define SWAP(x0,x) {float *tmp=x0;x0=x;x=tmp;}
// #define REFRESH_DELAY     10 //ms

dim3 grid, threads;

int dimX = 512;
int dimY = 512;
int size = 0;
int win_x = dimX;
int win_y = dimY;
int numVertices = (size * 2) / 4; // denominator is stride
int internalFormat = 4;

float dt = .1;
float diff = 0.00001f;
float visc = 0.000001f;
float force = 30.;
float buoy = 0.0;
float source_density = 2.0;
float source_temp = .25;
float dA = 0.0002; // diffusion constants
float dB = 0.00001;

float *chemA, *chemA_prev, *chemB, *chemB_prev, *laplacian;
float *vel[2], *vel_prev[2];
float *pressure, *pressure_prev;
float *temperature, *temperature_prev;
float *density, *density_prev;
float *divergence;
int *boundary;

float4 *displayPtr, *fboPtr, *displayPtr_d;
float2 *displayVertPtr;

GLuint  bufferObjDensity;
GLuint  textureID, vertexArrayID;
GLuint fboID, fboTxID, fboDepthTxID;
cudaGraphicsResource_t cgrTxData, cgrVertData;

Texture txDensity, txFbo;
FBO fboDisplay;

float avgFPS = 0.0f;
int fpsCount = 0;        // FPS count for averaging
int fpsLimit = 1;        // FPS limit for sampling
int frameNum = 0;
int animFrameNum = 0;
float framerate_sec = 1.0f/60.0f;

// StopWatchInterface *timer = NULL;
// timespec time1, time2;
// timespec time_diff(timespec start, timespec end);

// mouse controls
static int mouse_down[3];
int mouse_x, mouse_y, mouse_x_old, mouse_y_old;
bool togSimulate = false;
bool togDensity = true;
bool togVelocity = false;
bool togParticles = true;
bool togModBuoy = false;
bool hasRunOnce = false;
bool writeData = false;


void initVariables() {
  size = dimX * dimY;
  threads = dim3(16,16);
  grid.x = (dimX + threads.x - 1) / threads.x;
  grid.y = (dimY + threads.y - 1) / threads.y;

  displayPtr = (float4*)malloc(sizeof(float4)*size);
  displayVertPtr = (float2*)malloc(sizeof(float2)*numVertices);
  fboPtr = (float4*)malloc(sizeof(float4)*size);

  // writeData = argv[1];
  writeData = 0;

  // Create the CUTIL timer
  // sdkCreateTimer(&timer);
}

///////////////////////////////////////////////////////////////////////////////
// Initialize OpenGL
///////////////////////////////////////////////////////////////////////////////
void initGL() {
  // Framebuffer
  // glGenFramebuffersEXT(1, &fboID);
  // glBindFramebufferEXT(GL_FRAMEBUFFER, fboID);

  // Framebuffer's texture
  // glEnable(GL_TEXTURE_2D);
  // glGenTextures(1, &fboTxID);
  // glBindTexture(GL_TEXTURE_2D, fboTxID);
  // glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, win_x, win_y, 0, GL_RGBA, GL_FLOAT, 0);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  // glBindTexture(GL_TEXTURE_2D, 0);
  txFbo = Texture(win_x, win_y, Graphics::RGBA, Graphics::FLOAT);
  txFbo.submit();
  fboDisplay.attachTexture2D(txFbo.id(), FBO::COLOR_ATTACHMENT0);


  // The depth buffer
  // glGenRenderbuffersEXT(1, &fboDepthTxID);
  // glBindRenderbufferEXT(GL_RENDERBUFFER, fboDepthTxID);
  // glRenderbufferStorageEXT(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, win_x, win_y);
  // glFramebufferRenderbufferEXT(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fboDepthTxID);
  // glFramebufferTextureEXT(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, fboTxID, 0);

  // Set the list of draw buffers.
  // GLenum DrawBuffers[1] = {GL_COLOR_ATTACHMENT0};
  // glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

  // Set up a buffer object and bind texture to it
  // Later we register a cuda graphics resource to this in order to write to a texture
  glGenBuffers( 1, &bufferObjDensity );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, bufferObjDensity );
  glBufferData( GL_PIXEL_UNPACK_BUFFER_ARB, sizeof(float4) * dimX * dimY, NULL, GL_DYNAMIC_DRAW_ARB );
  glBindBuffer( GL_PIXEL_UNPACK_BUFFER_ARB, 0 );

  // Vertex buffer
  // Each vertex contains 3 floating point coordinates (x,y,z) and 4 color bytes (RGBA)
  // total 16 bytes per vertex
  glGenBuffers(1, &vertexArrayID);
  glBindBuffer( GL_ARRAY_BUFFER, vertexArrayID);
  glBufferData( GL_ARRAY_BUFFER, sizeof(float4)*numVertices, NULL, GL_DYNAMIC_DRAW_ARB );
  glBindBuffer( GL_ARRAY_BUFFER, 0 );

  // glGenTextures(1, &textureID);
  // glBindTexture(GL_TEXTURE_2D, textureID);
  // glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, dimX, dimY, 0, GL_BGRA, GL_FLOAT, NULL);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // glBindTexture(GL_TEXTURE_2D, 0);
  txDensity = Texture(dimX, dimY, Graphics::RGBA, Graphics::FLOAT);
  txDensity.submit();

  // Clean up
  glClearColor ( 0.0f, 0.0f, 0.0f, 1.0f );
	glClear ( GL_COLOR_BUFFER_BIT );

}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

static void pre_display ( void ) {
  // bind a framebuffer and render everything afterwards into that
  // glBindFramebufferEXT(GL_FRAMEBUFFER, fboID);
  // glViewport ( 0, 0, win_x, win_y );
  // glMatrixMode ( GL_PROJECTION );
  // glLoadIdentity ();
  // gluOrtho2D ( 0.0, 1.0, 0.0, 1.0 );
  glClearColor ( 0.2f, 0.2f, 0.2f, 1.0f );
  glClear(GL_COLOR_BUFFER_BIT);
}

static void post_display ( void ) {
  // unbind the framebuffer and draw its texture
  glBindFramebufferEXT(GL_FRAMEBUFFER, 0);

  glColor3f(1,1,1);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, fboTxID);
  glBegin(GL_QUADS);
    glTexCoord2f( 0, 1.0f);
    glVertex3f(0.0,1.0,0.0);
    glTexCoord2f(0,0);
    glVertex3f(0.0,0.0,0.0);
    glTexCoord2f(1.0f,0);
    glVertex3f(1.0f,0.0,0.0);
    glTexCoord2f(1.0f,1.0f);
    glVertex3f(1.0,1.0,0.0);
  glEnd();

  // glutSwapBuffers();

  // // now handle looping and writing data
  // if (time_diff(time1,time2).tv_nsec > framerate_sec) {
  //   if (togSimulate) {
  //     if (writeData) {
  //       glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, fboPtr);
  //       writeImage(outputImagePath, fboPtr, animFrameNum, win_x, win_y, internalFormat);
  //     }
  //     animFrameNum++;
  //     clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
  //     glutPostRedisplay(); // causes draw to loop forever
  //   }
  // }
}

void draw_density() {
  // glColor3f(1,1,1);
  glEnable(GL_TEXTURE_2D);
  txDensity.bind();
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, bufferObjDensity);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, dimX, dimY, GL_BGRA, GL_FLOAT, NULL); //sends the buffer

  glBegin(GL_QUADS);
    glTexCoord2f( 0, 1.0f);
    glVertex3f(0.0,1.0,0.0);
    glTexCoord2f(0,0);
    glVertex3f(0.0,0.0,0.0);
    glTexCoord2f(1.0f,0);
    glVertex3f(1.0f,0.0,0.0);
    glTexCoord2f(1.0f,1.0f);
    glVertex3f(1.0,1.0,0.0);
  glEnd();

  txDensity.unbind();
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  glDisable(GL_TEXTURE_2D);
}

class MyApp : public App{
public:


	MyApp(){
    // set fps to 200
		initWindow(Window::Dim(0,0, win_x, win_y), "CUDA Sim", 200);
	}

	virtual void onCreate(const ViewpointWindow& w){
    // initialize
    lens().near(0.1).far(500).fovy(45);
    nav().pos(Vec3f(.5,.5,1));

    initVariables();
    initGL();

    // Bind cudaGraphicsResource to GL buffer
    checkCudaErrors( cudaGraphicsGLRegisterBuffer(&cgrTxData, bufferObjDensity, cudaGraphicsMapFlagsWriteDiscard) );

    initGPUArrays();

    drawSquare(chemB, 1.0);

  }

	// ANIMATE
	virtual void onAnimate(double dt){

    // drawSquare(chemB, 1.0);

    dens_step( chemA, chemA_prev, chemB, chemB_prev, vel_prev[0], vel_prev[1], boundary, dt );

    makeColor(chemB, displayPtr_d);

    // if (frameNum > 0 && togSimulate) {
    //   get_from_UI(chemA_prev, chemB_prev, u_prev, v_prev);
    //   vel_step( u, v, u_prev, v_prev, chemB, visc, dt );
    //   dens_step( chemA, chemA_prev, chemB, chemB_prev, u, v, diff, dt );
    //   MakeColor<<<grid,threads>>>(chemB, displayPtr);
    //   MakeVerticesKernel<<<grid,threads>>>(displayVertPtr, u, v);
    // }
    //
    // getMappedPointer(displayPtr_d, cgrTxData);
    size_t  sizeT;
    cudaGraphicsMapResources( 1, &cgrTxData, 0 );
    cudaGraphicsResourceGetMappedPointer((void**)&displayPtr_d, &sizeT, cgrTxData);
    cudaGraphicsUnmapResources( 1, &cgrTxData, 0 );
    //
    // cudaGraphicsMapResources( 1, &cgrVertData, 0 );
    // checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&displayVertPtr, &sizeT, cgrVertData));
    // checkCudaErrors(cudaGraphicsUnmapResources( 1, &cgrVertData, 0 ));
    //
    // sdkStopTimer(&timer);
    // computeFPS();

	}

	// DRAW
	virtual void onDraw(Graphics& g, const Viewpoint& v){
    pre_display();

    draw_density();

    // post_display();
	}


	virtual void onKeyDown(const ViewpointWindow& w, const Keyboard& k){
    if (k.key()==' ') {
      cout << "cam pos: " << nav().pos() << endl;
      return;
    }
	}

	virtual void onMouseDown(const ViewpointWindow& w, const Mouse& m){

	}

	virtual void onMouseDrag(const ViewpointWindow& w, const Mouse& m){

	}

};


int main(){
	MyApp().start();
}
