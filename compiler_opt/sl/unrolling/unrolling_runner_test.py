import unittest
import sys
import datasets
import ray
import logging
from argparse import Namespace
from . import generate_unroll_results as gur
from . import generate_unroll_training_samples as guts
import com_pile_utils.generate_com_pile_loop_inputs as gcpli
import com_pile_utils.generate_com_pile_loop as gcpl
from com_pile_utils.dataset_writer import ID_FIELD

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

TEST_MODULE = b"""
; ModuleID = '/tmp/tmp40q172nj/1_intermediate_original_module'
source_filename = "/local-ssd/amdlibflame-otrkevhm552xspevawtkllysqp2goz3p-build/aidengro/spack-stage-amdlibflame-4.1-otrkevhm552xspevawtkllysqp2goz3p/spack-src/src/map/lapack2flamec/f2c/c/slatrs.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@.str = hidden unnamed_addr constant [2 x i8] c"U\00", align 1
@.str.1 = hidden unnamed_addr constant [2 x i8] c"N\00", align 1
@.str.2 = hidden unnamed_addr constant [2 x i8] c"L\00", align 1
@.str.3 = hidden unnamed_addr constant [2 x i8] c"T\00", align 1
@.str.4 = hidden unnamed_addr constant [2 x i8] c"C\00", align 1
@.str.5 = hidden unnamed_addr constant [2 x i8] c"Y\00", align 1
@.str.6 = hidden unnamed_addr constant [7 x i8] c"SLATRS\00", align 1
@.str.7 = hidden unnamed_addr constant [13 x i8] c"Safe minimum\00", align 1
@.str.8 = hidden unnamed_addr constant [10 x i8] c"Precision\00", align 1
@c__1 = external hidden global i32, align 4
@.str.9 = hidden unnamed_addr constant [9 x i8] c"Overflow\00", align 1
@.str.10 = hidden unnamed_addr constant [2 x i8] c"M\00", align 1
@c_b46 = external hidden global float, align 4

; Function Attrs: nounwind uwtable
declare i32 @slatrs_(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1

declare i32 @lsame_(ptr noundef, ptr noundef) #2

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

declare i32 @xerbla_(ptr noundef, ptr noundef) #2

declare float @slamch_(ptr noundef) #2

declare float @sasum_(ptr noundef, ptr noundef, ptr noundef) #2

declare i32 @isamax_(ptr noundef, ptr noundef, ptr noundef) #2

declare i32 @sscal_(ptr noundef, ptr noundef, ptr noundef, ptr noundef) #2

declare float @slange_(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) #2

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #3

declare i32 @strsv_(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) #2

declare i32 @saxpy_(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) #2

declare float @sdot_(ptr noundef, ptr noundef, ptr noundef, ptr noundef, ptr noundef) #2

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.0(ptr %i__1, ptr %i__2, i32 %0, ptr %add.ptr, ptr %incdec.ptr1) #0 {
newFuncRoot:
  br label %for.cond

for.cond:                                         ; preds = %newFuncRoot, %for.body
  %j.0 = phi i32 [ 1, %newFuncRoot ], [ %inc, %for.body ]
  %1 = load i32, ptr %i__1, align 4, !tbaa !4
  %cmp53.not = icmp sgt i32 %j.0, %1
  br i1 %cmp53.not, label %if.end79.loopexit.exitStub, label %for.body

for.body:                                         ; preds = %for.cond
  %sub54 = add nsw i32 %j.0, -1
  store i32 %sub54, ptr %i__2, align 4, !tbaa !4
  %mul = mul nsw i32 %j.0, %0
  %2 = sext i32 %mul to i64
  %3 = getelementptr float, ptr %add.ptr, i64 %2
  %arrayidx = getelementptr i8, ptr %3, i64 4
  %call56 = call float @sasum_(ptr noundef nonnull %i__2, ptr noundef %arrayidx, ptr noundef nonnull @c__1) #4
  %idxprom57 = zext nneg i32 %j.0 to i64
  %arrayidx58 = getelementptr inbounds nuw float, ptr %incdec.ptr1, i64 %idxprom57
  store float %call56, ptr %arrayidx58, align 4, !tbaa !8
  %inc = add nuw nsw i32 %j.0, 1
  br label %for.cond, !llvm.loop !10

if.end79.loopexit.exitStub:                       ; preds = %for.cond
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.1(ptr %i__1, ptr %n, ptr %i__2, i32 %0, ptr %add.ptr, ptr %incdec.ptr1) #0 {
newFuncRoot:
  br label %for.cond61

for.cond61:                                       ; preds = %newFuncRoot, %for.body63
  %j.1 = phi i32 [ 1, %newFuncRoot ], [ %inc74, %for.body63 ]
  %1 = load i32, ptr %i__1, align 4, !tbaa !4
  %cmp62.not = icmp sgt i32 %j.1, %1
  br i1 %cmp62.not, label %for.end75.exitStub, label %for.body63

for.body63:                                       ; preds = %for.cond61
  %2 = load i32, ptr %n, align 4, !tbaa !4
  %sub64 = sub nsw i32 %2, %j.1
  store i32 %sub64, ptr %i__2, align 4, !tbaa !4
  %add65 = add nuw nsw i32 %j.1, 1
  %mul66 = mul nsw i32 %j.1, %0
  %add67 = add nsw i32 %add65, %mul66
  %idxprom68 = sext i32 %add67 to i64
  %arrayidx69 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom68
  %call70 = call float @sasum_(ptr noundef nonnull %i__2, ptr noundef %arrayidx69, ptr noundef nonnull @c__1) #4
  %idxprom71 = zext nneg i32 %j.1 to i64
  %arrayidx72 = getelementptr inbounds nuw float, ptr %incdec.ptr1, i64 %idxprom71
  store float %call70, ptr %arrayidx72, align 4, !tbaa !8
  %inc74 = add nuw nsw i32 %j.1, 1
  br label %for.cond61, !llvm.loop !12

for.end75.exitStub:                               ; preds = %for.cond61
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.2(ptr %i__1, ptr %i__2, i32 %0, ptr %add.ptr, ptr %sumj, ptr %r__1, ptr %tmax.0.out) #0 {
newFuncRoot:
  br label %for.cond97

for.cond97:                                       ; preds = %newFuncRoot, %for.body99
  %tmax.0 = phi float [ 0.000000e+00, %newFuncRoot ], [ %call105.tmax.0, %for.body99 ]
  %j.2 = phi i32 [ 2, %newFuncRoot ], [ %inc115, %for.body99 ]
  store float %tmax.0, ptr %tmax.0.out, align 4
  %1 = load i32, ptr %i__1, align 4, !tbaa !4
  %cmp98.not = icmp sgt i32 %j.2, %1
  br i1 %cmp98.not, label %if.end140.loopexit511.exitStub, label %for.body99

for.body99:                                       ; preds = %for.cond97
  %sub100 = add nsw i32 %j.2, -1
  store i32 %sub100, ptr %i__2, align 4, !tbaa !4
  %mul101 = mul nsw i32 %j.2, %0
  %2 = sext i32 %mul101 to i64
  %3 = getelementptr float, ptr %add.ptr, i64 %2
  %arrayidx104 = getelementptr i8, ptr %3, i64 4
  %call105 = call float @slange_(ptr noundef nonnull @.str.10, ptr noundef nonnull %i__2, ptr noundef nonnull @c__1, ptr noundef %arrayidx104, ptr noundef nonnull @c__1, ptr noundef nonnull %sumj) #4
  store float %call105, ptr %r__1, align 4, !tbaa !8
  %cmp109 = fcmp ogt float %call105, %tmax.0
  %call105.tmax.0 = select i1 %cmp109, float %call105, float %tmax.0
  %inc115 = add nuw nsw i32 %j.2, 1
  br label %for.cond97, !llvm.loop !13

if.end140.loopexit511.exitStub:                   ; preds = %for.cond97
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.3(ptr %i__1, ptr %n, ptr %i__2, i32 %0, ptr %add.ptr, ptr %sumj, ptr %r__1, ptr %tmax.1.out) #0 {
newFuncRoot:
  br label %for.cond119

for.cond119:                                      ; preds = %newFuncRoot, %for.body121
  %tmax.1 = phi float [ 0.000000e+00, %newFuncRoot ], [ %call128.tmax.1, %for.body121 ]
  %j.3 = phi i32 [ 1, %newFuncRoot ], [ %inc138, %for.body121 ]
  store float %tmax.1, ptr %tmax.1.out, align 4
  %1 = load i32, ptr %i__1, align 4, !tbaa !4
  %cmp120.not = icmp sgt i32 %j.3, %1
  br i1 %cmp120.not, label %if.end140.loopexit.exitStub, label %for.body121

for.body121:                                      ; preds = %for.cond119
  %2 = load i32, ptr %n, align 4, !tbaa !4
  %sub122 = sub nsw i32 %2, %j.3
  store i32 %sub122, ptr %i__2, align 4, !tbaa !4
  %add123 = add nuw nsw i32 %j.3, 1
  %mul124 = mul nsw i32 %j.3, %0
  %add125 = add nsw i32 %add123, %mul124
  %idxprom126 = sext i32 %add125 to i64
  %arrayidx127 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom126
  %call128 = call float @slange_(ptr noundef nonnull @.str.10, ptr noundef nonnull %i__2, ptr noundef nonnull @c__1, ptr noundef %arrayidx127, ptr noundef nonnull @c__1, ptr noundef nonnull %sumj) #4
  store float %call128, ptr %r__1, align 4, !tbaa !8
  %cmp132 = fcmp ogt float %call128, %tmax.1
  %call128.tmax.1 = select i1 %cmp132, float %call128, float %tmax.1
  %inc138 = add nuw nsw i32 %j.3, 1
  br label %for.cond119, !llvm.loop !14

if.end140.loopexit.exitStub:                      ; preds = %for.cond119
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.4(ptr %i__1, ptr %incdec.ptr1, ptr %tscal, i32 %call, ptr %i__2, i32 %0, ptr %add.ptr, ptr %r__1, ptr %n) #0 {
newFuncRoot:
  br label %for.cond146

for.cond146:                                      ; preds = %newFuncRoot, %for.inc204
  %j.4 = phi i32 [ 1, %newFuncRoot ], [ %inc205, %for.inc204 ]
  %1 = load i32, ptr %i__1, align 4, !tbaa !4
  %cmp147.not = icmp sgt i32 %j.4, %1
  br i1 %cmp147.not, label %if.end214.loopexit.exitStub, label %for.body148

for.body148:                                      ; preds = %for.cond146
  %idxprom149 = zext nneg i32 %j.4 to i64
  %arrayidx150 = getelementptr inbounds nuw float, ptr %incdec.ptr1, i64 %idxprom149
  %2 = load float, ptr %arrayidx150, align 4, !tbaa !8
  %call151 = call float @slamch_(ptr noundef nonnull @.str.9) #4
  %cmp152 = fcmp ugt float %2, %call151
  br i1 %cmp152, label %if.else157, label %if.then153

if.then153:                                       ; preds = %for.body148
  %3 = load float, ptr %tscal, align 4, !tbaa !8
  %idxprom154 = zext nneg i32 %j.4 to i64
  %arrayidx155 = getelementptr inbounds nuw float, ptr %incdec.ptr1, i64 %idxprom154
  %4 = load float, ptr %arrayidx155, align 4, !tbaa !8
  %mul156 = fmul float %4, %3
  store float %mul156, ptr %arrayidx155, align 4, !tbaa !8
  br label %for.inc204

if.else157:                                       ; preds = %for.body148
  %idxprom158 = zext nneg i32 %j.4 to i64
  %arrayidx159 = getelementptr inbounds nuw float, ptr %incdec.ptr1, i64 %idxprom158
  store float 0.000000e+00, ptr %arrayidx159, align 4, !tbaa !8
  %tobool160.not = icmp eq i32 %call, 0
  br i1 %tobool160.not, label %if.else181, label %if.then161

if.then161:                                       ; preds = %if.else157
  %sub162 = add nsw i32 %j.4, -1
  store i32 %sub162, ptr %i__2, align 4, !tbaa !4
  br label %codeRepl

codeRepl:                                         ; preds = %if.then161
  call void @__llvm_extracted_loop.5(ptr %i__2, ptr %tscal, i32 %j.4, i32 %0, ptr %add.ptr, ptr %r__1, ptr %incdec.ptr1)
  br label %for.inc204.loopexit510

for.inc204.loopexit510:                           ; preds = %codeRepl
  br label %for.inc204

if.else181:                                       ; preds = %if.else157
  %5 = load i32, ptr %n, align 4, !tbaa !4
  store i32 %5, ptr %i__2, align 4, !tbaa !4
  br label %codeRepl1

codeRepl1:                                        ; preds = %if.else181
  call void @__llvm_extracted_loop.6(i32 %j.4, ptr %i__2, ptr %tscal, i32 %0, ptr %add.ptr, ptr %r__1, ptr %incdec.ptr1)
  br label %for.inc204.loopexit

for.inc204.loopexit:                              ; preds = %codeRepl1
  br label %for.inc204

for.inc204:                                       ; preds = %for.inc204.loopexit510, %for.inc204.loopexit, %if.then153
  %inc205 = add nuw nsw i32 %j.4, 1
  br label %for.cond146, !llvm.loop !15

if.end214.loopexit.exitStub:                      ; preds = %for.cond146
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.5(ptr %i__2, ptr %tscal, i32 %j.4, i32 %0, ptr %add.ptr, ptr %r__1, ptr %incdec.ptr1) #0 {
newFuncRoot:
  br label %for.cond163

for.cond163:                                      ; preds = %newFuncRoot, %for.body165
  %i__.0 = phi i32 [ 1, %newFuncRoot ], [ %inc179, %for.body165 ]
  %1 = load i32, ptr %i__2, align 4, !tbaa !4
  %cmp164.not = icmp sgt i32 %i__.0, %1
  br i1 %cmp164.not, label %for.inc204.loopexit510.exitStub, label %for.body165

for.body165:                                      ; preds = %for.cond163
  %2 = load float, ptr %tscal, align 4, !tbaa !8
  %mul166 = mul nsw i32 %j.4, %0
  %add167 = add nsw i32 %i__.0, %mul166
  %idxprom168 = sext i32 %add167 to i64
  %arrayidx169 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom168
  %3 = load float, ptr %arrayidx169, align 4, !tbaa !8
  store float %3, ptr %r__1, align 4, !tbaa !8
  %cmp170 = fcmp ult float %3, 0.000000e+00
  %4 = load float, ptr %r__1, align 4
  %5 = load float, ptr %r__1, align 4
  %fneg = fneg float %5
  %cond174 = select i1 %cmp170, float %fneg, float %4
  %idxprom176 = zext nneg i32 %j.4 to i64
  %arrayidx177 = getelementptr inbounds nuw float, ptr %incdec.ptr1, i64 %idxprom176
  %6 = load float, ptr %arrayidx177, align 4, !tbaa !8
  %7 = call float @llvm.fmuladd.f32(float %2, float %cond174, float %6)
  store float %7, ptr %arrayidx177, align 4, !tbaa !8
  %inc179 = add nuw nsw i32 %i__.0, 1
  br label %for.cond163, !llvm.loop !16

for.inc204.loopexit510.exitStub:                  ; preds = %for.cond163
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.6(i32 %j.4, ptr %i__2, ptr %tscal, i32 %0, ptr %add.ptr, ptr %r__1, ptr %incdec.ptr1) #0 {
newFuncRoot:
  br label %for.cond183

for.cond183:                                      ; preds = %newFuncRoot, %for.body185
  %i__.1.in = phi i32 [ %j.4, %newFuncRoot ], [ %i__.1, %for.body185 ]
  %i__.1 = add nuw nsw i32 %i__.1.in, 1
  %1 = load i32, ptr %i__2, align 4, !tbaa !4
  %cmp184.not.not = icmp slt i32 %i__.1.in, %1
  br i1 %cmp184.not.not, label %for.body185, label %for.inc204.loopexit.exitStub

for.body185:                                      ; preds = %for.cond183
  %2 = load float, ptr %tscal, align 4, !tbaa !8
  %mul186 = mul nsw i32 %j.4, %0
  %add187 = add nsw i32 %i__.1, %mul186
  %idxprom188 = sext i32 %add187 to i64
  %arrayidx189 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom188
  %3 = load float, ptr %arrayidx189, align 4, !tbaa !8
  store float %3, ptr %r__1, align 4, !tbaa !8
  %cmp190 = fcmp ult float %3, 0.000000e+00
  %4 = load float, ptr %r__1, align 4
  %5 = load float, ptr %r__1, align 4
  %fneg193 = fneg float %5
  %cond195 = select i1 %cmp190, float %fneg193, float %4
  %idxprom197 = zext nneg i32 %j.4 to i64
  %arrayidx198 = getelementptr inbounds nuw float, ptr %incdec.ptr1, i64 %idxprom197
  %6 = load float, ptr %arrayidx198, align 4, !tbaa !8
  %7 = call float @llvm.fmuladd.f32(float %2, float %cond195, float %6)
  store float %7, ptr %arrayidx198, align 4, !tbaa !8
  br label %for.cond183, !llvm.loop !17

for.inc204.loopexit.exitStub:                     ; preds = %for.cond183
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.7(float %div244, i32 %.493, ptr %i__2, ptr %i__1, float %div, i32 %0, ptr %add.ptr, ptr %r__1, ptr %incdec.ptr1, ptr %grow.9.ph.ce.out) #0 {
newFuncRoot:
  br label %for.cond245

for.cond245:                                      ; preds = %newFuncRoot, %for.inc303
  %grow.0 = phi float [ %div244, %newFuncRoot ], [ %grow.1, %for.inc303 ]
  %xbnd.0 = phi float [ %div244, %newFuncRoot ], [ %cond289, %for.inc303 ]
  %j.5 = phi i32 [ %.493, %newFuncRoot ], [ %add304, %for.inc303 ]
  %1 = load i32, ptr %i__2, align 4, !tbaa !4
  %cmp246 = icmp slt i32 %1, 0
  %2 = load i32, ptr %i__1, align 4
  %cmp248 = icmp sge i32 %j.5, %2
  %3 = load i32, ptr %i__1, align 4
  %cmp250 = icmp sle i32 %j.5, %3
  %cond253.in = select i1 %cmp246, i1 %cmp248, i1 %cmp250
  br i1 %cond253.in, label %for.body255, label %if.end477.loopexit509.split

for.body255:                                      ; preds = %for.cond245
  %cmp256 = fcmp ugt float %grow.0, %div
  br i1 %cmp256, label %if.end259, label %if.end477.loopexit509.split

if.end259:                                        ; preds = %for.body255
  %mul260489 = add i32 %0, 1
  %add261 = mul i32 %j.5, %mul260489
  %idxprom262 = sext i32 %add261 to i64
  %arrayidx263 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom262
  %4 = load float, ptr %arrayidx263, align 4, !tbaa !8
  store float %4, ptr %r__1, align 4, !tbaa !8
  %cmp264 = fcmp ult float %4, 0.000000e+00
  %5 = load float, ptr %r__1, align 4
  %6 = load float, ptr %r__1, align 4
  %fneg268 = fneg float %6
  %cond270 = select i1 %cmp264, float %fneg268, float %5
  store float %xbnd.0, ptr %r__1, align 4, !tbaa !8
  %cmp274 = fcmp ogt float %cond270, 1.000000e+00
  %cond279 = select i1 %cmp274, float 1.000000e+00, float %cond270
  %mul280 = fmul float %cond279, %grow.0
  %7 = load float, ptr %r__1, align 4, !tbaa !8
  %cmp284 = fcmp olt float %7, %mul280
  %cond289 = select i1 %cmp284, float %7, float %mul280
  %idxprom290 = sext i32 %j.5 to i64
  %arrayidx291 = getelementptr inbounds float, ptr %incdec.ptr1, i64 %idxprom290
  %8 = load float, ptr %arrayidx291, align 4, !tbaa !8
  %add292 = fadd float %cond270, %8
  %cmp293 = fcmp ult float %add292, %div
  br i1 %cmp293, label %for.inc303, label %if.then295

if.then295:                                       ; preds = %if.end259
  %idxprom296 = sext i32 %j.5 to i64
  %arrayidx297 = getelementptr inbounds float, ptr %incdec.ptr1, i64 %idxprom296
  %9 = load float, ptr %arrayidx297, align 4, !tbaa !8
  %add298 = fadd float %cond270, %9
  %div299 = fdiv float %cond270, %add298
  %mul300 = fmul float %grow.0, %div299
  br label %for.inc303

for.inc303:                                       ; preds = %if.then295, %if.end259
  %grow.1 = phi float [ %mul300, %if.then295 ], [ 0.000000e+00, %if.end259 ]
  %10 = load i32, ptr %i__2, align 4, !tbaa !4
  %add304 = add nsw i32 %j.5, %10
  br label %for.cond245, !llvm.loop !18

if.end477.loopexit509.split:                      ; preds = %for.body255, %for.cond245
  %grow.9.ph.ce = phi float [ %xbnd.0, %for.cond245 ], [ %grow.0, %for.body255 ]
  store float %grow.9.ph.ce, ptr %grow.9.ph.ce.out, align 4
  br label %if.end477.loopexit509.exitStub

if.end477.loopexit509.exitStub:                   ; preds = %if.end477.loopexit509.split
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.8(float %cond325, i32 %.493, ptr %i__1, ptr %i__2, float %div, ptr %incdec.ptr1, ptr %grow.2.out) #0 {
newFuncRoot:
  br label %for.cond326

for.cond326:                                      ; preds = %newFuncRoot, %for.inc348
  %grow.2 = phi float [ %cond325, %newFuncRoot ], [ %mul347, %for.inc348 ]
  %j.6 = phi i32 [ %.493, %newFuncRoot ], [ %add349, %for.inc348 ]
  store float %grow.2, ptr %grow.2.out, align 4
  %0 = load i32, ptr %i__1, align 4, !tbaa !4
  %cmp327 = icmp slt i32 %0, 0
  %1 = load i32, ptr %i__2, align 4
  %cmp330 = icmp sge i32 %j.6, %1
  %2 = load i32, ptr %i__2, align 4
  %cmp333 = icmp sle i32 %j.6, %2
  %cond336.in = select i1 %cmp327, i1 %cmp330, i1 %cmp333
  %cmp339 = fcmp ugt float %grow.2, %div
  %or.cond = select i1 %cond336.in, i1 %cmp339, i1 false
  br i1 %or.cond, label %for.inc348, label %if.end477.loopexit508.exitStub

for.inc348:                                       ; preds = %for.cond326
  %idxprom343 = sext i32 %j.6 to i64
  %arrayidx344 = getelementptr inbounds float, ptr %incdec.ptr1, i64 %idxprom343
  %3 = load float, ptr %arrayidx344, align 4, !tbaa !8
  %add345 = fadd float %3, 1.000000e+00
  %div346 = fdiv float 1.000000e+00, %add345
  %mul347 = fmul float %grow.2, %div346
  %4 = load i32, ptr %i__1, align 4, !tbaa !4
  %add349 = add nsw i32 %j.6, %4
  br label %for.cond326, !llvm.loop !19

if.end477.loopexit508.exitStub:                   ; preds = %for.cond326
  ret void
}

; Function Attrs: nounwind uwtable
define hidden i1 @__llvm_extracted_loop.9(float %div372, i32 %.497, ptr %i__2, ptr %i__1, float %div, ptr %incdec.ptr1, ptr %r__1, i32 %0, ptr %add.ptr, ptr %grow.5.out, ptr %xbnd.1.out) #0 {
newFuncRoot:
  br label %for.cond373

for.cond373:                                      ; preds = %newFuncRoot, %for.inc420
  %grow.5 = phi float [ %div372, %newFuncRoot ], [ %grow.5.div393, %for.inc420 ]
  %xbnd.1 = phi float [ %div372, %newFuncRoot ], [ %xbnd.2, %for.inc420 ]
  %j.7 = phi i32 [ %.497, %newFuncRoot ], [ %add421, %for.inc420 ]
  store float %xbnd.1, ptr %xbnd.1.out, align 4
  store float %grow.5, ptr %grow.5.out, align 4
  %1 = load i32, ptr %i__2, align 4, !tbaa !4
  %cmp374 = icmp slt i32 %1, 0
  %2 = load i32, ptr %i__1, align 4
  %cmp377 = icmp sge i32 %j.7, %2
  %3 = load i32, ptr %i__1, align 4
  %cmp380 = icmp sle i32 %j.7, %3
  %cond383.in = select i1 %cmp374, i1 %cmp377, i1 %cmp380
  br i1 %cond383.in, label %for.body385, label %for.end422.exitStub

for.body385:                                      ; preds = %for.cond373
  %cmp386 = fcmp ugt float %grow.5, %div
  br i1 %cmp386, label %if.end389, label %if.end477.loopexit507.exitStub

if.end389:                                        ; preds = %for.body385
  %idxprom390 = sext i32 %j.7 to i64
  %arrayidx391 = getelementptr inbounds float, ptr %incdec.ptr1, i64 %idxprom390
  %4 = load float, ptr %arrayidx391, align 4, !tbaa !8
  %add392 = fadd float %4, 1.000000e+00
  store float %grow.5, ptr %r__1, align 4, !tbaa !8
  %div393 = fdiv float %xbnd.1, %add392
  %cmp397 = fcmp olt float %grow.5, %div393
  %grow.5.div393 = select i1 %cmp397, float %grow.5, float %div393
  %mul403488 = add i32 %0, 1
  %add404 = mul i32 %j.7, %mul403488
  %idxprom405 = sext i32 %add404 to i64
  %arrayidx406 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom405
  %5 = load float, ptr %arrayidx406, align 4, !tbaa !8
  store float %5, ptr %r__1, align 4, !tbaa !8
  %cmp407 = fcmp ult float %5, 0.000000e+00
  %6 = load float, ptr %r__1, align 4
  %7 = load float, ptr %r__1, align 4
  %fneg411 = fneg float %7
  %cond413 = select i1 %cmp407, float %fneg411, float %6
  %cmp414 = fcmp ogt float %add392, %cond413
  br i1 %cmp414, label %if.then416, label %for.inc420

if.then416:                                       ; preds = %if.end389
  %div417 = fdiv float %cond413, %add392
  %mul418 = fmul float %xbnd.1, %div417
  br label %for.inc420

for.inc420:                                       ; preds = %if.end389, %if.then416
  %xbnd.2 = phi float [ %mul418, %if.then416 ], [ %xbnd.1, %if.end389 ]
  %8 = load i32, ptr %i__2, align 4, !tbaa !4
  %add421 = add nsw i32 %j.7, %8
  br label %for.cond373, !llvm.loop !20

for.end422.exitStub:                              ; preds = %for.cond373
  ret i1 true

if.end477.loopexit507.exitStub:                   ; preds = %for.body385
  ret i1 false
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.10(float %cond451, i32 %.497, ptr %i__1, ptr %i__2, float %div, ptr %incdec.ptr1, ptr %grow.6.out) #0 {
newFuncRoot:
  br label %for.cond452

for.cond452:                                      ; preds = %newFuncRoot, %for.inc473
  %grow.6 = phi float [ %cond451, %newFuncRoot ], [ %div472, %for.inc473 ]
  %j.8 = phi i32 [ %.497, %newFuncRoot ], [ %add474, %for.inc473 ]
  store float %grow.6, ptr %grow.6.out, align 4
  %0 = load i32, ptr %i__1, align 4, !tbaa !4
  %cmp453 = icmp slt i32 %0, 0
  %1 = load i32, ptr %i__2, align 4
  %cmp456 = icmp sge i32 %j.8, %1
  %2 = load i32, ptr %i__2, align 4
  %cmp459 = icmp sle i32 %j.8, %2
  %cond462.in = select i1 %cmp453, i1 %cmp456, i1 %cmp459
  %cmp465 = fcmp ugt float %grow.6, %div
  %or.cond502 = select i1 %cond462.in, i1 %cmp465, i1 false
  br i1 %or.cond502, label %for.inc473, label %if.end477.loopexit.exitStub

for.inc473:                                       ; preds = %for.cond452
  %idxprom469 = sext i32 %j.8 to i64
  %arrayidx470 = getelementptr inbounds float, ptr %incdec.ptr1, i64 %idxprom469
  %3 = load float, ptr %arrayidx470, align 4, !tbaa !8
  %add471 = fadd float %3, 1.000000e+00
  %div472 = fdiv float %grow.6, %add471
  %4 = load i32, ptr %i__1, align 4, !tbaa !4
  %add474 = add nsw i32 %j.8, %4
  br label %for.cond452, !llvm.loop !21

if.end477.loopexit.exitStub:                      ; preds = %for.cond452
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.11(float %xmax.0, i32 %jfirst.2, ptr %i__2, ptr %i__1, ptr %incdec.ptr, ptr %r__1, i32 %call3, i32 %0, ptr %add.ptr, ptr %tscal, float %div, ptr %n, ptr %i__3, ptr %scale, float %div47, ptr %rec, ptr %incdec.ptr1, ptr %x, i32 %call) #0 {
newFuncRoot:
  br label %for.cond496

for.cond496:                                      ; preds = %newFuncRoot, %for.inc710
  %xmax.1 = phi float [ %xmax.0, %newFuncRoot ], [ %xmax.10, %for.inc710 ]
  %j.9 = phi i32 [ %jfirst.2, %newFuncRoot ], [ %add711, %for.inc710 ]
  %1 = load i32, ptr %i__2, align 4, !tbaa !4
  %cmp497 = icmp slt i32 %1, 0
  %2 = load i32, ptr %i__1, align 4
  %cmp500 = icmp sge i32 %j.9, %2
  %3 = load i32, ptr %i__1, align 4
  %cmp503 = icmp sle i32 %j.9, %3
  %cond506.in = select i1 %cmp497, i1 %cmp500, i1 %cmp503
  br i1 %cond506.in, label %for.body508, label %if.end980.loopexit506.exitStub

for.body508:                                      ; preds = %for.cond496
  %idxprom509 = sext i32 %j.9 to i64
  %arrayidx510 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom509
  %4 = load float, ptr %arrayidx510, align 4, !tbaa !8
  store float %4, ptr %r__1, align 4, !tbaa !8
  %cmp511 = fcmp ult float %4, 0.000000e+00
  %5 = load float, ptr %r__1, align 4
  %6 = load float, ptr %r__1, align 4
  %fneg515 = fneg float %6
  %cond517 = select i1 %cmp511, float %fneg515, float %5
  %tobool518.not = icmp eq i32 %call3, 0
  br i1 %tobool518.not, label %if.else525, label %if.then519

if.then519:                                       ; preds = %for.body508
  %mul520492 = add i32 %0, 1
  %add521 = mul i32 %j.9, %mul520492
  %idxprom522 = sext i32 %add521 to i64
  %arrayidx523 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom522
  %7 = load float, ptr %arrayidx523, align 4, !tbaa !8
  %8 = load float, ptr %tscal, align 4, !tbaa !8
  %mul524 = fmul float %7, %8
  br label %if.end530

if.else525:                                       ; preds = %for.body508
  %9 = load float, ptr %tscal, align 4, !tbaa !8
  %cmp526 = fcmp oeq float %9, 1.000000e+00
  br i1 %cmp526, label %L95, label %if.end529

if.end529:                                        ; preds = %if.else525
  %10 = load float, ptr %tscal, align 4, !tbaa !8
  br label %if.end530

if.end530:                                        ; preds = %if.end529, %if.then519
  %tjjs.0 = phi float [ %mul524, %if.then519 ], [ %10, %if.end529 ]
  %cmp531 = fcmp ult float %tjjs.0, 0.000000e+00
  %fneg535 = fneg float %tjjs.0
  %cond537 = select i1 %cmp531, float %fneg535, float %tjjs.0
  %cmp538 = fcmp ogt float %cond537, %div
  br i1 %cmp538, label %if.then540, label %if.else567

if.else567:                                       ; preds = %if.end530
  %cmp568 = fcmp ogt float %cond537, 0.000000e+00
  br i1 %cmp568, label %if.then570, label %if.else603

if.else603:                                       ; preds = %if.else567
  %11 = load i32, ptr %n, align 4, !tbaa !4
  store i32 %11, ptr %i__3, align 4, !tbaa !4
  br label %codeRepl

codeRepl:                                         ; preds = %if.else603
  call void @__llvm_extracted_loop.12(ptr %i__3, ptr %incdec.ptr)
  br label %for.end612

for.end612:                                       ; preds = %codeRepl
  %idxprom613 = sext i32 %j.9 to i64
  %arrayidx614 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom613
  store float 1.000000e+00, ptr %arrayidx614, align 4, !tbaa !8
  store float 0.000000e+00, ptr %scale, align 4, !tbaa !8
  br label %L95

if.then570:                                       ; preds = %if.else567
  %mul571 = fmul float %cond537, %div47
  %cmp572 = fcmp ogt float %cond517, %mul571
  br i1 %cmp572, label %if.then574, label %if.end590

if.then574:                                       ; preds = %if.then570
  %mul575 = fmul float %cond537, %div47
  %div576 = fdiv float %mul575, %cond517
  store float %div576, ptr %rec, align 4, !tbaa !8
  %idxprom577 = sext i32 %j.9 to i64
  %arrayidx578 = getelementptr inbounds float, ptr %incdec.ptr1, i64 %idxprom577
  %12 = load float, ptr %arrayidx578, align 4, !tbaa !8
  %cmp579 = fcmp ogt float %12, 1.000000e+00
  br i1 %cmp579, label %if.then581, label %if.end585

if.then581:                                       ; preds = %if.then574
  %idxprom582 = sext i32 %j.9 to i64
  %arrayidx583 = getelementptr inbounds float, ptr %incdec.ptr1, i64 %idxprom582
  %13 = load float, ptr %arrayidx583, align 4, !tbaa !8
  %14 = load float, ptr %rec, align 4, !tbaa !8
  %div584 = fdiv float %14, %13
  store float %div584, ptr %rec, align 4, !tbaa !8
  br label %if.end585

if.end585:                                        ; preds = %if.then581, %if.then574
  %call587 = call i32 @sscal_(ptr noundef nonnull %n, ptr noundef nonnull %rec, ptr noundef %x, ptr noundef nonnull @c__1) #4
  %15 = load float, ptr %rec, align 4, !tbaa !8
  %16 = load float, ptr %scale, align 4, !tbaa !8
  %mul588 = fmul float %16, %15
  store float %mul588, ptr %scale, align 4, !tbaa !8
  %mul589 = fmul float %xmax.1, %15
  br label %if.end590

if.end590:                                        ; preds = %if.end585, %if.then570
  %xmax.4 = phi float [ %mul589, %if.end585 ], [ %xmax.1, %if.then570 ]
  %idxprom591 = sext i32 %j.9 to i64
  %arrayidx592 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom591
  %17 = load float, ptr %arrayidx592, align 4, !tbaa !8
  %div593 = fdiv float %17, %tjjs.0
  store float %div593, ptr %arrayidx592, align 4, !tbaa !8
  %idxprom594 = sext i32 %j.9 to i64
  %arrayidx595 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom594
  %18 = load float, ptr %arrayidx595, align 4, !tbaa !8
  store float %18, ptr %r__1, align 4, !tbaa !8
  %cmp596 = fcmp ult float %18, 0.000000e+00
  %19 = load float, ptr %r__1, align 4
  %20 = load float, ptr %r__1, align 4
  %fneg600 = fneg float %20
  %cond602 = select i1 %cmp596, float %fneg600, float %19
  br label %L95

if.then540:                                       ; preds = %if.end530
  %cmp541 = fcmp olt float %cond537, 1.000000e+00
  %mul544 = fmul float %cond537, %div47
  %cmp545 = fcmp ogt float %cond517, %mul544
  %or.cond503 = select i1 %cmp541, i1 %cmp545, i1 false
  br i1 %or.cond503, label %if.then547, label %if.end554

if.then547:                                       ; preds = %if.then540
  %div548 = fdiv float 1.000000e+00, %cond517
  store float %div548, ptr %rec, align 4, !tbaa !8
  %call550 = call i32 @sscal_(ptr noundef nonnull %n, ptr noundef nonnull %rec, ptr noundef %x, ptr noundef nonnull @c__1) #4
  %21 = load float, ptr %rec, align 4, !tbaa !8
  %22 = load float, ptr %scale, align 4, !tbaa !8
  %mul551 = fmul float %22, %21
  store float %mul551, ptr %scale, align 4, !tbaa !8
  %mul552 = fmul float %xmax.1, %21
  br label %if.end554

if.end554:                                        ; preds = %if.then547, %if.then540
  %xmax.3 = phi float [ %xmax.1, %if.then540 ], [ %mul552, %if.then547 ]
  %idxprom555 = sext i32 %j.9 to i64
  %arrayidx556 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom555
  %23 = load float, ptr %arrayidx556, align 4, !tbaa !8
  %div557 = fdiv float %23, %tjjs.0
  store float %div557, ptr %arrayidx556, align 4, !tbaa !8
  %idxprom558 = sext i32 %j.9 to i64
  %arrayidx559 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom558
  %24 = load float, ptr %arrayidx559, align 4, !tbaa !8
  store float %24, ptr %r__1, align 4, !tbaa !8
  %cmp560 = fcmp ult float %24, 0.000000e+00
  %25 = load float, ptr %r__1, align 4
  %26 = load float, ptr %r__1, align 4
  %fneg564 = fneg float %26
  %cond566 = select i1 %cmp560, float %fneg564, float %25
  br label %L95

L95:                                              ; preds = %if.end554, %for.end612, %if.end590, %if.else525
  %xmax.7 = phi float [ %xmax.1, %if.else525 ], [ %xmax.3, %if.end554 ], [ %xmax.4, %if.end590 ], [ 0.000000e+00, %for.end612 ]
  %xj.2 = phi float [ %cond517, %if.else525 ], [ %cond566, %if.end554 ], [ %cond602, %if.end590 ], [ 1.000000e+00, %for.end612 ]
  %cmp617 = fcmp ogt float %xj.2, 1.000000e+00
  br i1 %cmp617, label %if.then619, label %if.else633

if.else633:                                       ; preds = %L95
  %idxprom634 = sext i32 %j.9 to i64
  %arrayidx635 = getelementptr inbounds float, ptr %incdec.ptr1, i64 %idxprom634
  %27 = load float, ptr %arrayidx635, align 4, !tbaa !8
  %mul636 = fmul float %xj.2, %27
  %sub637 = fsub float %div47, %xmax.7
  %cmp638 = fcmp ogt float %mul636, %sub637
  br i1 %cmp638, label %if.then640, label %if.end645

if.then640:                                       ; preds = %if.else633
  %call642 = call i32 @sscal_(ptr noundef nonnull %n, ptr noundef nonnull @c_b46, ptr noundef %x, ptr noundef nonnull @c__1) #4
  %28 = load float, ptr %scale, align 4, !tbaa !8
  %mul643 = fmul float %28, 5.000000e-01
  store float %mul643, ptr %scale, align 4, !tbaa !8
  br label %if.end645

if.then619:                                       ; preds = %L95
  %div620 = fdiv float 1.000000e+00, %xj.2
  store float %div620, ptr %rec, align 4, !tbaa !8
  %idxprom621 = sext i32 %j.9 to i64
  %arrayidx622 = getelementptr inbounds float, ptr %incdec.ptr1, i64 %idxprom621
  %29 = load float, ptr %arrayidx622, align 4, !tbaa !8
  %sub623 = fsub float %div47, %xmax.7
  %mul624 = fmul float %sub623, %div620
  %cmp625 = fcmp ogt float %29, %mul624
  br i1 %cmp625, label %if.then627, label %if.end645

if.then627:                                       ; preds = %if.then619
  %30 = load float, ptr %rec, align 4, !tbaa !8
  %mul628 = fmul float %30, 5.000000e-01
  store float %mul628, ptr %rec, align 4, !tbaa !8
  %call630 = call i32 @sscal_(ptr noundef nonnull %n, ptr noundef nonnull %rec, ptr noundef %x, ptr noundef nonnull @c__1) #4
  %31 = load float, ptr %rec, align 4, !tbaa !8
  %32 = load float, ptr %scale, align 4, !tbaa !8
  %mul631 = fmul float %32, %31
  store float %mul631, ptr %scale, align 4, !tbaa !8
  br label %if.end645

if.end645:                                        ; preds = %if.else633, %if.then640, %if.then619, %if.then627
  %tobool646.not = icmp eq i32 %call, 0
  br i1 %tobool646.not, label %if.else675, label %if.then647

if.then647:                                       ; preds = %if.end645
  %cmp648 = icmp sgt i32 %j.9, 1
  br i1 %cmp648, label %if.then650, label %for.inc710

if.then650:                                       ; preds = %if.then647
  %sub651 = add nsw i32 %j.9, -1
  store i32 %sub651, ptr %i__3, align 4, !tbaa !4
  %idxprom652 = zext nneg i32 %j.9 to i64
  %arrayidx653 = getelementptr inbounds nuw float, ptr %incdec.ptr, i64 %idxprom652
  %33 = load float, ptr %arrayidx653, align 4, !tbaa !8
  %fneg654 = fneg float %33
  %34 = load float, ptr %tscal, align 4, !tbaa !8
  %mul655 = fmul float %34, %fneg654
  store float %mul655, ptr %r__1, align 4, !tbaa !8
  %mul656 = mul nsw i32 %j.9, %0
  %35 = sext i32 %mul656 to i64
  %36 = getelementptr float, ptr %add.ptr, i64 %35
  %arrayidx659 = getelementptr i8, ptr %36, i64 4
  %call661 = call i32 @saxpy_(ptr noundef nonnull %i__3, ptr noundef nonnull %r__1, ptr noundef %arrayidx659, ptr noundef nonnull @c__1, ptr noundef %x, ptr noundef nonnull @c__1) #4
  %sub662 = add nsw i32 %j.9, -1
  store i32 %sub662, ptr %i__3, align 4, !tbaa !4
  %call664 = call i32 @isamax_(ptr noundef nonnull %i__3, ptr noundef %x, ptr noundef nonnull @c__1) #4
  %idxprom665 = sext i32 %call664 to i64
  %arrayidx666 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom665
  %37 = load float, ptr %arrayidx666, align 4, !tbaa !8
  store float %37, ptr %r__1, align 4, !tbaa !8
  %cmp667 = fcmp ult float %37, 0.000000e+00
  %38 = load float, ptr %r__1, align 4
  %39 = load float, ptr %r__1, align 4
  %fneg671 = fneg float %39
  %cond673 = select i1 %cmp667, float %fneg671, float %38
  br label %for.inc710

if.else675:                                       ; preds = %if.end645
  %40 = load i32, ptr %n, align 4, !tbaa !4
  %cmp676 = icmp slt i32 %j.9, %40
  br i1 %cmp676, label %if.then678, label %for.inc710

if.then678:                                       ; preds = %if.else675
  %41 = load i32, ptr %n, align 4, !tbaa !4
  %sub679 = sub nsw i32 %41, %j.9
  store i32 %sub679, ptr %i__3, align 4, !tbaa !4
  %idxprom680 = sext i32 %j.9 to i64
  %arrayidx681 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom680
  %42 = load float, ptr %arrayidx681, align 4, !tbaa !8
  %fneg682 = fneg float %42
  %43 = load float, ptr %tscal, align 4, !tbaa !8
  %mul683 = fmul float %43, %fneg682
  store float %mul683, ptr %r__1, align 4, !tbaa !8
  %add684 = add nsw i32 %j.9, 1
  %mul685 = mul nsw i32 %j.9, %0
  %add686 = add nsw i32 %add684, %mul685
  %idxprom687 = sext i32 %add686 to i64
  %arrayidx688 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom687
  %44 = sext i32 %j.9 to i64
  %45 = getelementptr float, ptr %incdec.ptr, i64 %44
  %arrayidx691 = getelementptr i8, ptr %45, i64 4
  %call692 = call i32 @saxpy_(ptr noundef nonnull %i__3, ptr noundef nonnull %r__1, ptr noundef %arrayidx688, ptr noundef nonnull @c__1, ptr noundef %arrayidx691, ptr noundef nonnull @c__1) #4
  %46 = load i32, ptr %n, align 4, !tbaa !4
  %sub693 = sub nsw i32 %46, %j.9
  store i32 %sub693, ptr %i__3, align 4, !tbaa !4
  %47 = sext i32 %j.9 to i64
  %48 = getelementptr float, ptr %incdec.ptr, i64 %47
  %arrayidx696 = getelementptr i8, ptr %48, i64 4
  %call697 = call i32 @isamax_(ptr noundef nonnull %i__3, ptr noundef %arrayidx696, ptr noundef nonnull @c__1) #4
  %add698 = add nsw i32 %j.9, %call697
  %idxprom699 = sext i32 %add698 to i64
  %arrayidx700 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom699
  %49 = load float, ptr %arrayidx700, align 4, !tbaa !8
  store float %49, ptr %r__1, align 4, !tbaa !8
  %cmp701 = fcmp ult float %49, 0.000000e+00
  %50 = load float, ptr %r__1, align 4
  %51 = load float, ptr %r__1, align 4
  %fneg705 = fneg float %51
  %cond707 = select i1 %cmp701, float %fneg705, float %50
  br label %for.inc710

for.inc710:                                       ; preds = %if.then650, %if.then647, %if.then678, %if.else675
  %xmax.10 = phi float [ %cond673, %if.then650 ], [ %xmax.7, %if.then647 ], [ %cond707, %if.then678 ], [ %xmax.7, %if.else675 ]
  %52 = load i32, ptr %i__2, align 4, !tbaa !4
  %add711 = add nsw i32 %j.9, %52
  br label %for.cond496, !llvm.loop !22

if.end980.loopexit506.exitStub:                   ; preds = %for.cond496
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.12(ptr %i__3, ptr %incdec.ptr) #0 {
newFuncRoot:
  br label %for.cond604

for.cond604:                                      ; preds = %newFuncRoot, %for.body607
  %i__.2 = phi i32 [ 1, %newFuncRoot ], [ %inc611, %for.body607 ]
  %0 = load i32, ptr %i__3, align 4, !tbaa !4
  %cmp605.not = icmp sgt i32 %i__.2, %0
  br i1 %cmp605.not, label %for.end612.exitStub, label %for.body607

for.body607:                                      ; preds = %for.cond604
  %idxprom608 = zext nneg i32 %i__.2 to i64
  %arrayidx609 = getelementptr inbounds nuw float, ptr %incdec.ptr, i64 %idxprom608
  store float 0.000000e+00, ptr %arrayidx609, align 4, !tbaa !8
  %inc611 = add nuw nsw i32 %i__.2, 1
  br label %for.cond604, !llvm.loop !23

for.end612.exitStub:                              ; preds = %for.cond604
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.13(float %xmax.0, i32 %jfirst.2, ptr %i__1, ptr %i__2, ptr %incdec.ptr, ptr %r__1, ptr %tscal, ptr %rec, ptr %incdec.ptr1, float %div47, i32 %call3, i32 %0, ptr %add.ptr, ptr %n, ptr %x, ptr %scale, ptr %sumj, i32 %call, ptr %i__3, float %div) #0 {
newFuncRoot:
  br label %for.cond714

for.cond714:                                      ; preds = %newFuncRoot, %if.end958
  %xmax.11 = phi float [ %xmax.0, %newFuncRoot ], [ %cond976, %if.end958 ]
  %tjjs.1 = phi float [ undef, %newFuncRoot ], [ %tjjs.6, %if.end958 ]
  %j.10 = phi i32 [ %jfirst.2, %newFuncRoot ], [ %add978, %if.end958 ]
  %1 = load i32, ptr %i__1, align 4, !tbaa !4
  %cmp715 = icmp slt i32 %1, 0
  %2 = load i32, ptr %i__2, align 4
  %cmp718 = icmp sge i32 %j.10, %2
  %3 = load i32, ptr %i__2, align 4
  %cmp721 = icmp sle i32 %j.10, %3
  %cond724.in = select i1 %cmp715, i1 %cmp718, i1 %cmp721
  br i1 %cond724.in, label %for.body726, label %if.end980.loopexit.exitStub

for.body726:                                      ; preds = %for.cond714
  %idxprom727 = sext i32 %j.10 to i64
  %arrayidx728 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom727
  %4 = load float, ptr %arrayidx728, align 4, !tbaa !8
  store float %4, ptr %r__1, align 4, !tbaa !8
  %cmp729 = fcmp ult float %4, 0.000000e+00
  %5 = load float, ptr %r__1, align 4
  %6 = load float, ptr %r__1, align 4
  %fneg733 = fneg float %6
  %cond735 = select i1 %cmp729, float %fneg733, float %5
  %7 = load float, ptr %tscal, align 4, !tbaa !8
  %cmp739 = fcmp ogt float %xmax.11, 1.000000e+00
  %cond744 = select i1 %cmp739, float %xmax.11, float 1.000000e+00
  %div745 = fdiv float 1.000000e+00, %cond744
  store float %div745, ptr %rec, align 4, !tbaa !8
  %idxprom746 = sext i32 %j.10 to i64
  %arrayidx747 = getelementptr inbounds float, ptr %incdec.ptr1, i64 %idxprom746
  %8 = load float, ptr %arrayidx747, align 4, !tbaa !8
  %sub748 = fsub float %div47, %cond735
  %mul749 = fmul float %sub748, %div745
  %cmp750 = fcmp ogt float %8, %mul749
  br i1 %cmp750, label %if.then752, label %if.end793

if.then752:                                       ; preds = %for.body726
  %9 = load float, ptr %rec, align 4, !tbaa !8
  %mul753 = fmul float %9, 5.000000e-01
  store float %mul753, ptr %rec, align 4, !tbaa !8
  %tobool754.not = icmp eq i32 %call3, 0
  br i1 %tobool754.not, label %if.else761, label %if.then755

if.then755:                                       ; preds = %if.then752
  %mul756490 = add i32 %0, 1
  %add757 = mul i32 %j.10, %mul756490
  %idxprom758 = sext i32 %add757 to i64
  %arrayidx759 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom758
  %10 = load float, ptr %arrayidx759, align 4, !tbaa !8
  %11 = load float, ptr %tscal, align 4, !tbaa !8
  %mul760 = fmul float %10, %11
  br label %if.end762

if.else761:                                       ; preds = %if.then752
  %12 = load float, ptr %tscal, align 4, !tbaa !8
  br label %if.end762

if.end762:                                        ; preds = %if.else761, %if.then755
  %tjjs.2 = phi float [ %mul760, %if.then755 ], [ %12, %if.else761 ]
  %cmp763 = fcmp ult float %tjjs.2, 0.000000e+00
  %fneg767 = fneg float %tjjs.2
  %cond769 = select i1 %cmp763, float %fneg767, float %tjjs.2
  %cmp770 = fcmp ogt float %cond769, 1.000000e+00
  br i1 %cmp770, label %if.then772, label %if.end784

if.then772:                                       ; preds = %if.end762
  store float 1.000000e+00, ptr %r__1, align 4, !tbaa !8
  %13 = load float, ptr %rec, align 4, !tbaa !8
  %mul773 = fmul float %13, %cond769
  %cmp777 = fcmp ogt float %mul773, 1.000000e+00
  %.mul773 = select i1 %cmp777, float 1.000000e+00, float %mul773
  store float %.mul773, ptr %rec, align 4, !tbaa !8
  %div783 = fdiv float %7, %tjjs.2
  br label %if.end784

if.end784:                                        ; preds = %if.then772, %if.end762
  %uscal.0 = phi float [ %div783, %if.then772 ], [ %7, %if.end762 ]
  %14 = load float, ptr %rec, align 4, !tbaa !8
  %cmp785 = fcmp olt float %14, 1.000000e+00
  br i1 %cmp785, label %if.then787, label %if.end793

if.then787:                                       ; preds = %if.end784
  %call789 = call i32 @sscal_(ptr noundef nonnull %n, ptr noundef nonnull %rec, ptr noundef %x, ptr noundef nonnull @c__1) #4
  %15 = load float, ptr %rec, align 4, !tbaa !8
  %16 = load float, ptr %scale, align 4, !tbaa !8
  %mul790 = fmul float %16, %15
  store float %mul790, ptr %scale, align 4, !tbaa !8
  %mul791 = fmul float %xmax.11, %15
  br label %if.end793

if.end793:                                        ; preds = %if.end784, %if.then787, %for.body726
  %uscal.1 = phi float [ %7, %for.body726 ], [ %uscal.0, %if.then787 ], [ %uscal.0, %if.end784 ]
  %xmax.13 = phi float [ %xmax.11, %for.body726 ], [ %mul791, %if.then787 ], [ %xmax.11, %if.end784 ]
  %tjjs.3 = phi float [ %tjjs.1, %for.body726 ], [ %tjjs.2, %if.then787 ], [ %tjjs.2, %if.end784 ]
  store float 0.000000e+00, ptr %sumj, align 4, !tbaa !8
  %cmp794 = fcmp oeq float %uscal.1, 1.000000e+00
  br i1 %cmp794, label %if.then796, label %if.else822

if.else822:                                       ; preds = %if.end793
  %tobool823.not = icmp eq i32 %call, 0
  br i1 %tobool823.not, label %if.else841, label %if.then824

if.then824:                                       ; preds = %if.else822
  %sub825 = add nsw i32 %j.10, -1
  store i32 %sub825, ptr %i__3, align 4, !tbaa !4
  br label %codeRepl

codeRepl:                                         ; preds = %if.then824
  call void @__llvm_extracted_loop.14(ptr %i__3, i32 %j.10, i32 %0, ptr %add.ptr, float %uscal.1, ptr %incdec.ptr, ptr %sumj)
  br label %if.end863.loopexit505

if.end863.loopexit505:                            ; preds = %codeRepl
  br label %if.end863

if.else841:                                       ; preds = %if.else822
  %17 = load i32, ptr %n, align 4, !tbaa !4
  %cmp842 = icmp slt i32 %j.10, %17
  br i1 %cmp842, label %if.then844, label %if.end863

if.then844:                                       ; preds = %if.else841
  %18 = load i32, ptr %n, align 4, !tbaa !4
  store i32 %18, ptr %i__3, align 4, !tbaa !4
  br label %codeRepl1

codeRepl1:                                        ; preds = %if.then844
  call void @__llvm_extracted_loop.15(i32 %j.10, ptr %i__3, i32 %0, ptr %add.ptr, float %uscal.1, ptr %incdec.ptr, ptr %sumj)
  br label %if.end863.loopexit

if.end863.loopexit:                               ; preds = %codeRepl1
  br label %if.end863

if.then796:                                       ; preds = %if.end793
  %tobool797.not = icmp eq i32 %call, 0
  br i1 %tobool797.not, label %if.else806, label %if.then798

if.then798:                                       ; preds = %if.then796
  %sub799 = add nsw i32 %j.10, -1
  store i32 %sub799, ptr %i__3, align 4, !tbaa !4
  %mul800 = mul nsw i32 %j.10, %0
  %19 = sext i32 %mul800 to i64
  %20 = getelementptr float, ptr %add.ptr, i64 %19
  %arrayidx803 = getelementptr i8, ptr %20, i64 4
  %call805 = call float @sdot_(ptr noundef nonnull %i__3, ptr noundef %arrayidx803, ptr noundef nonnull @c__1, ptr noundef %x, ptr noundef nonnull @c__1) #4
  store float %call805, ptr %sumj, align 4, !tbaa !8
  br label %if.end863

if.else806:                                       ; preds = %if.then796
  %21 = load i32, ptr %n, align 4, !tbaa !4
  %cmp807 = icmp slt i32 %j.10, %21
  br i1 %cmp807, label %if.then809, label %if.end863

if.then809:                                       ; preds = %if.else806
  %22 = load i32, ptr %n, align 4, !tbaa !4
  %sub810 = sub nsw i32 %22, %j.10
  store i32 %sub810, ptr %i__3, align 4, !tbaa !4
  %add811 = add nsw i32 %j.10, 1
  %mul812 = mul nsw i32 %j.10, %0
  %add813 = add nsw i32 %add811, %mul812
  %idxprom814 = sext i32 %add813 to i64
  %arrayidx815 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom814
  %23 = sext i32 %j.10 to i64
  %24 = getelementptr float, ptr %incdec.ptr, i64 %23
  %arrayidx818 = getelementptr i8, ptr %24, i64 4
  %call819 = call float @sdot_(ptr noundef nonnull %i__3, ptr noundef %arrayidx815, ptr noundef nonnull @c__1, ptr noundef %arrayidx818, ptr noundef nonnull @c__1) #4
  store float %call819, ptr %sumj, align 4, !tbaa !8
  br label %if.end863

if.end863:                                        ; preds = %if.end863.loopexit505, %if.end863.loopexit, %if.else841, %if.then798, %if.then809, %if.else806
  %25 = load float, ptr %tscal, align 4, !tbaa !8
  %cmp864 = fcmp oeq float %uscal.1, %25
  br i1 %cmp864, label %if.then866, label %if.else951

if.else951:                                       ; preds = %if.end863
  %idxprom952 = sext i32 %j.10 to i64
  %arrayidx953 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom952
  %26 = load float, ptr %arrayidx953, align 4, !tbaa !8
  %div954 = fdiv float %26, %tjjs.3
  %27 = load float, ptr %sumj, align 4, !tbaa !8
  %sub955 = fsub float %div954, %27
  %idxprom956 = sext i32 %j.10 to i64
  %arrayidx957 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom956
  store float %sub955, ptr %arrayidx957, align 4, !tbaa !8
  br label %if.end958

if.then866:                                       ; preds = %if.end863
  %28 = load float, ptr %sumj, align 4, !tbaa !8
  %idxprom867 = sext i32 %j.10 to i64
  %arrayidx868 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom867
  %29 = load float, ptr %arrayidx868, align 4, !tbaa !8
  %sub869 = fsub float %29, %28
  store float %sub869, ptr %arrayidx868, align 4, !tbaa !8
  %idxprom870 = sext i32 %j.10 to i64
  %arrayidx871 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom870
  %30 = load float, ptr %arrayidx871, align 4, !tbaa !8
  store float %30, ptr %r__1, align 4, !tbaa !8
  %cmp872 = fcmp ult float %30, 0.000000e+00
  %31 = load float, ptr %r__1, align 4
  %32 = load float, ptr %r__1, align 4
  %fneg876 = fneg float %32
  %cond878 = select i1 %cmp872, float %fneg876, float %31
  %tobool879.not = icmp eq i32 %call3, 0
  br i1 %tobool879.not, label %if.else886, label %if.then880

if.then880:                                       ; preds = %if.then866
  %mul881491 = add i32 %0, 1
  %add882 = mul i32 %j.10, %mul881491
  %idxprom883 = sext i32 %add882 to i64
  %arrayidx884 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom883
  %33 = load float, ptr %arrayidx884, align 4, !tbaa !8
  %34 = load float, ptr %tscal, align 4, !tbaa !8
  %mul885 = fmul float %33, %34
  br label %if.end891

if.else886:                                       ; preds = %if.then866
  %35 = load float, ptr %tscal, align 4, !tbaa !8
  %cmp887 = fcmp oeq float %35, 1.000000e+00
  br i1 %cmp887, label %if.end958, label %if.end891

if.end891:                                        ; preds = %if.else886, %if.then880
  %tjjs.4 = phi float [ %mul885, %if.then880 ], [ %35, %if.else886 ]
  %cmp892 = fcmp ult float %tjjs.4, 0.000000e+00
  %fneg896 = fneg float %tjjs.4
  %cond898 = select i1 %cmp892, float %fneg896, float %tjjs.4
  %cmp899 = fcmp ogt float %cond898, %div
  br i1 %cmp899, label %if.then901, label %if.else919

if.else919:                                       ; preds = %if.end891
  %cmp920 = fcmp ogt float %cond898, 0.000000e+00
  br i1 %cmp920, label %if.then922, label %if.else937

if.else937:                                       ; preds = %if.else919
  %36 = load i32, ptr %n, align 4, !tbaa !4
  store i32 %36, ptr %i__3, align 4, !tbaa !4
  br label %codeRepl2

codeRepl2:                                        ; preds = %if.else937
  call void @__llvm_extracted_loop.16(ptr %i__3, ptr %incdec.ptr)
  br label %for.end946

for.end946:                                       ; preds = %codeRepl2
  %idxprom947 = sext i32 %j.10 to i64
  %arrayidx948 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom947
  store float 1.000000e+00, ptr %arrayidx948, align 4, !tbaa !8
  store float 0.000000e+00, ptr %scale, align 4, !tbaa !8
  br label %if.end958

if.then922:                                       ; preds = %if.else919
  %mul923 = fmul float %cond898, %div47
  %cmp924 = fcmp ogt float %cond878, %mul923
  br i1 %cmp924, label %if.then926, label %if.end933

if.then926:                                       ; preds = %if.then922
  %mul927 = fmul float %cond898, %div47
  %div928 = fdiv float %mul927, %cond878
  store float %div928, ptr %rec, align 4, !tbaa !8
  %call930 = call i32 @sscal_(ptr noundef nonnull %n, ptr noundef nonnull %rec, ptr noundef %x, ptr noundef nonnull @c__1) #4
  %37 = load float, ptr %rec, align 4, !tbaa !8
  %38 = load float, ptr %scale, align 4, !tbaa !8
  %mul931 = fmul float %38, %37
  store float %mul931, ptr %scale, align 4, !tbaa !8
  %mul932 = fmul float %xmax.13, %37
  br label %if.end933

if.end933:                                        ; preds = %if.then926, %if.then922
  %xmax.16 = phi float [ %mul932, %if.then926 ], [ %xmax.13, %if.then922 ]
  %idxprom934 = sext i32 %j.10 to i64
  %arrayidx935 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom934
  %39 = load float, ptr %arrayidx935, align 4, !tbaa !8
  %div936 = fdiv float %39, %tjjs.4
  store float %div936, ptr %arrayidx935, align 4, !tbaa !8
  br label %if.end958

if.then901:                                       ; preds = %if.end891
  %cmp902 = fcmp olt float %cond898, 1.000000e+00
  %mul905 = fmul float %cond898, %div47
  %cmp906 = fcmp ogt float %cond878, %mul905
  %or.cond504 = select i1 %cmp902, i1 %cmp906, i1 false
  br i1 %or.cond504, label %if.then908, label %if.end915

if.then908:                                       ; preds = %if.then901
  %div909 = fdiv float 1.000000e+00, %cond878
  store float %div909, ptr %rec, align 4, !tbaa !8
  %call911 = call i32 @sscal_(ptr noundef nonnull %n, ptr noundef nonnull %rec, ptr noundef %x, ptr noundef nonnull @c__1) #4
  %40 = load float, ptr %rec, align 4, !tbaa !8
  %41 = load float, ptr %scale, align 4, !tbaa !8
  %mul912 = fmul float %41, %40
  store float %mul912, ptr %scale, align 4, !tbaa !8
  %mul913 = fmul float %xmax.13, %40
  br label %if.end915

if.end915:                                        ; preds = %if.then908, %if.then901
  %xmax.15 = phi float [ %xmax.13, %if.then901 ], [ %mul913, %if.then908 ]
  %idxprom916 = sext i32 %j.10 to i64
  %arrayidx917 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom916
  %42 = load float, ptr %arrayidx917, align 4, !tbaa !8
  %div918 = fdiv float %42, %tjjs.4
  store float %div918, ptr %arrayidx917, align 4, !tbaa !8
  br label %if.end958

if.end958:                                        ; preds = %if.else886, %if.end933, %for.end946, %if.end915, %if.else951
  %xmax.20 = phi float [ %xmax.13, %if.else951 ], [ %xmax.13, %if.else886 ], [ %xmax.15, %if.end915 ], [ %xmax.16, %if.end933 ], [ 0.000000e+00, %for.end946 ]
  %tjjs.6 = phi float [ %tjjs.3, %if.else951 ], [ %35, %if.else886 ], [ %tjjs.4, %if.end933 ], [ %tjjs.4, %for.end946 ], [ %tjjs.4, %if.end915 ]
  %idxprom959 = sext i32 %j.10 to i64
  %arrayidx960 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom959
  %43 = load float, ptr %arrayidx960, align 4, !tbaa !8
  store float %43, ptr %r__1, align 4, !tbaa !8
  %cmp961 = fcmp ult float %43, 0.000000e+00
  %44 = load float, ptr %r__1, align 4
  %45 = load float, ptr %r__1, align 4
  %fneg965 = fneg float %45
  %cond967 = select i1 %cmp961, float %fneg965, float %44
  %cmp971 = fcmp ogt float %xmax.20, %cond967
  %cond976 = select i1 %cmp971, float %xmax.20, float %cond967
  %46 = load i32, ptr %i__1, align 4, !tbaa !4
  %add978 = add nsw i32 %j.10, %46
  br label %for.cond714, !llvm.loop !24

if.end980.loopexit.exitStub:                      ; preds = %for.cond714
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.14(ptr %i__3, i32 %j.10, i32 %0, ptr %add.ptr, float %uscal.1, ptr %incdec.ptr, ptr %sumj) #0 {
newFuncRoot:
  br label %for.cond826

for.cond826:                                      ; preds = %newFuncRoot, %for.body829
  %i__.3 = phi i32 [ 1, %newFuncRoot ], [ %inc839, %for.body829 ]
  %1 = load i32, ptr %i__3, align 4, !tbaa !4
  %cmp827.not = icmp sgt i32 %i__.3, %1
  br i1 %cmp827.not, label %if.end863.loopexit505.exitStub, label %for.body829

for.body829:                                      ; preds = %for.cond826
  %mul830 = mul nsw i32 %j.10, %0
  %add831 = add nsw i32 %i__.3, %mul830
  %idxprom832 = sext i32 %add831 to i64
  %arrayidx833 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom832
  %2 = load float, ptr %arrayidx833, align 4, !tbaa !8
  %mul834 = fmul float %2, %uscal.1
  %idxprom835 = zext nneg i32 %i__.3 to i64
  %arrayidx836 = getelementptr inbounds nuw float, ptr %incdec.ptr, i64 %idxprom835
  %3 = load float, ptr %arrayidx836, align 4, !tbaa !8
  %4 = load float, ptr %sumj, align 4, !tbaa !8
  %5 = call float @llvm.fmuladd.f32(float %mul834, float %3, float %4)
  store float %5, ptr %sumj, align 4, !tbaa !8
  %inc839 = add nuw nsw i32 %i__.3, 1
  br label %for.cond826, !llvm.loop !25

if.end863.loopexit505.exitStub:                   ; preds = %for.cond826
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.15(i32 %j.10, ptr %i__3, i32 %0, ptr %add.ptr, float %uscal.1, ptr %incdec.ptr, ptr %sumj) #0 {
newFuncRoot:
  br label %for.cond846

for.cond846:                                      ; preds = %newFuncRoot, %for.body849
  %i__.4.in = phi i32 [ %j.10, %newFuncRoot ], [ %i__.4, %for.body849 ]
  %i__.4 = add nsw i32 %i__.4.in, 1
  %1 = load i32, ptr %i__3, align 4, !tbaa !4
  %cmp847.not.not = icmp slt i32 %i__.4.in, %1
  br i1 %cmp847.not.not, label %for.body849, label %if.end863.loopexit.exitStub

for.body849:                                      ; preds = %for.cond846
  %mul850 = mul nsw i32 %j.10, %0
  %add851 = add nsw i32 %i__.4, %mul850
  %idxprom852 = sext i32 %add851 to i64
  %arrayidx853 = getelementptr inbounds float, ptr %add.ptr, i64 %idxprom852
  %2 = load float, ptr %arrayidx853, align 4, !tbaa !8
  %mul854 = fmul float %2, %uscal.1
  %idxprom855 = sext i32 %i__.4 to i64
  %arrayidx856 = getelementptr inbounds float, ptr %incdec.ptr, i64 %idxprom855
  %3 = load float, ptr %arrayidx856, align 4, !tbaa !8
  %4 = load float, ptr %sumj, align 4, !tbaa !8
  %5 = call float @llvm.fmuladd.f32(float %mul854, float %3, float %4)
  store float %5, ptr %sumj, align 4, !tbaa !8
  br label %for.cond846, !llvm.loop !26

if.end863.loopexit.exitStub:                      ; preds = %for.cond846
  ret void
}

; Function Attrs: nounwind uwtable
define hidden void @__llvm_extracted_loop.16(ptr %i__3, ptr %incdec.ptr) #0 {
newFuncRoot:
  br label %for.cond938

for.cond938:                                      ; preds = %newFuncRoot, %for.body941
  %i__.5 = phi i32 [ 1, %newFuncRoot ], [ %inc945, %for.body941 ]
  %0 = load i32, ptr %i__3, align 4, !tbaa !4
  %cmp939.not = icmp sgt i32 %i__.5, %0
  br i1 %cmp939.not, label %for.end946.exitStub, label %for.body941

for.body941:                                      ; preds = %for.cond938
  %idxprom942 = zext nneg i32 %i__.5 to i64
  %arrayidx943 = getelementptr inbounds nuw float, ptr %incdec.ptr, i64 %idxprom942
  store float 0.000000e+00, ptr %arrayidx943, align 4, !tbaa !8
  %inc945 = add nuw nsw i32 %i__.5, 1
  br label %for.cond938, !llvm.loop !27

for.end946.exitStub:                              ; preds = %for.cond938
  ret void
}

attributes #0 = { nounwind uwtable "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="znver2" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+avx,+avx2,+cmov,+crc32,+cx8,+fma,+fxsr,+mmx,+popcnt,+sse,+sse2,+sse3,+sse4.1,+sse4.2,+ssse3,+x87,+xsave" "tune-cpu"="znver2" }
attributes #3 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0, !1, !2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"uwtable", i32 2}
!3 = !{!"clang version 18.0.0 (https://github.com/llvm-ml/llvm-project b452eb491a2ae09c12cc88b715f003377cec543b)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C/C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"float", !6, i64 0}
!10 = distinct !{!10, !11}
!11 = !{!"llvm.loop.mustprogress"}
!12 = distinct !{!12, !11}
!13 = distinct !{!13, !11}
!14 = distinct !{!14, !11}
!15 = distinct !{!15, !11}
!16 = distinct !{!16, !11}
!17 = distinct !{!17, !11}
!18 = distinct !{!18, !11}
!19 = distinct !{!19, !11}
!20 = distinct !{!20, !11}
!21 = distinct !{!21, !11}
!22 = distinct !{!22, !11}
!23 = distinct !{!23, !11}
!24 = distinct !{!24, !11}
!25 = distinct !{!25, !11}
!26 = distinct !{!26, !11}
!27 = distinct !{!27, !11}
"""

TEST_MODULE2 = b"""
define dso_local void @_Z8vec_initPdi(ptr noundef captures(none) %a, i32 noundef %n) local_unnamed_addr #0 {
entry0:
  %cmp1 = icmp sle i32 %n, 10000
  br i1 %cmp1, label %entry, label %for.cond.cleanup
entry:
  %cmp9 = icmp sgt i32 %n, 0
  br i1 %cmp9, label %for.body.preheader, label %for.cond.cleanup

for.body.preheader:                               ; preds = %entry
  %wide.trip.count = zext nneg i32 %n to i64
  br label %for.body

for.cond.cleanup:                                 ; preds = %for.body, %entry
  ret void

for.body:                                         ; preds = %for.body.preheader, %for.body
  %indvars.iv = phi i64 [ 0, %for.body.preheader ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds nuw double, ptr %a, i64 %indvars.iv
  store double 3.010400e+04, ptr %arrayidx, align 8
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond.not = icmp eq i64 %indvars.iv.next, %wide.trip.count
  br i1 %exitcond.not, label %for.cond.cleanup, label %for.body
}
attributes #0 = { inputgen_entry }
"""


if True:
    SAVE_TEMPS = True
    TEMP_DIR = "./temps"
    DEBUG = True
else:
    SAVE_TEMPS = False
    TEMP_DIR = None
    DEBUG = False


class GenCompileLoopInputsTest(unittest.TestCase):
    def test_input_gen2(self):
        loop = dict()
        loop["module"] = TEST_MODULE
        loop["num_loops"] = 17
        loop["language"] = "c"
        loop[ID_FIELD] = 15

        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=SAVE_TEMPS,
            temp_dir=TEMP_DIR,
            debug_instrumentation=False,
        )
        loop_inputs = gcpli.process_module(args, 13, loop)
        logger.debug(loop_inputs)
        self.assertIsNotNone(loop_inputs)
        self.assertEqual(loop_inputs.i, 13)
        logger.debug(loop_inputs.data)
        logger.debug(loop_inputs.data[0])
        logger.debug(loop_inputs.data[0]["inputs"])
        self.assertGreater(len(loop_inputs.data[0]["inputs"]), 0)

        loop = loop_inputs.data[0]
        loop[ID_FIELD] = 17

        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=SAVE_TEMPS,
            temp_dir=TEMP_DIR,
            dump_llvm=False,
            debug=DEBUG,
        )
        res = gur.process_module(args, 14, loop)
        print(res)
        loop = res.data[0]
        loop[ID_FIELD] = 18

        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=SAVE_TEMPS,
            temp_dir=TEMP_DIR,
            dump_llvm=False,
            debug=DEBUG,
        )
        res = guts.process_module(args, 9, loop)
        print(res)

    def test_input_gen(self):
        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=SAVE_TEMPS,
            temp_dir=TEMP_DIR,
        )
        data = dict()
        data["content"] = TEST_MODULE
        data["language"] = "c"
        data[ID_FIELD] = 15
        loops = gcpl.process_module(args, 10, data)
        self.assertIsNotNone(loops)
        logger.debug("loops")
        logger.debug(loops)
        self.assertEqual(loops.i, 10)
        ds = loops.data
        logger.debug(ds)
        loop = ds[0]
        logger.debug("loop")
        logger.debug(loop)
        loop[ID_FIELD] = 16

        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=True,
            temp_dir="./temps1/",
            # save_temps=False,
            # temp_dir=None,
            debug_instrumentation=False,
        )
        loop_inputs = gcpli.process_module(args, 13, loop)
        logger.debug(loop_inputs)
        self.assertIsNotNone(loop_inputs)
        self.assertEqual(loop_inputs.i, 13)
        logger.debug(loop_inputs.data)
        logger.debug(loop_inputs.data[0])
        logger.debug(loop_inputs.data[0]["inputs"])
        self.assertGreater(len(loop_inputs.data[0]["inputs"]), 0)

        loop = loop_inputs.data[0]
        loop[ID_FIELD] = 17

        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=True,
            temp_dir="./temps2/",
            # save_temps=False,
            # temp_dir=None,
            dump_llvm=False,
            debug=True,
        )
        res = gur.process_module(args, 14, loop)
        print(res)
        loop = res.data[0]
        loop[ID_FIELD] = 18

        args = Namespace(
            mclang=[],
            mllvm=[],
            save_temps=True,
            temp_dir="./temps3/",
            dump_llvm=False,
            debug=False,
        )
        res = guts.process_module(args, 9, loop)
        print(res)


if __name__ == "__main__":
    unittest.main()
