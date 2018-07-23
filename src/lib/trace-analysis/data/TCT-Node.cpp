// -*-Mode: C++;-*-

// * BeginRiceCopyright *****************************************************
//
// $HeadURL$
// $Id$
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2002-2017, Rice University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage.
//
// ******************************************************* EndRiceCopyright *

/* 
 * File:   TCT-Node.cpp
 * Author: Lai Wei <lai.wei@rice.edu>
 *
 * Created on March 5, 2018, 3:08 PM
 * 
 * Temporal Context Tree nodes.
 */

#include "TCT-Node.hpp"
#include "../DifferenceQuantifier.hpp"

namespace TraceAnalysis {
  string TCTANode::toString(int maxDepth, Time minDuration) const {
    string ret;
    /*
    ret += "CFG-0x";
    if (cfgGraph == NULL) ret += vmaToHexString(0);
    else ret += vmaToHexString(cfgGraph->vma);
    ret += " RA-0x" + vmaToHexString(ra) + " ";
    */
    for (int i = 0; i < depth; i++) ret += "  ";
    ret += name + id.toString();
    ret += " " + time.toString();
    ret += ", " + std::to_string(getDuration()/getSamplingPeriod()) + " samples";
    if (weight > 1) ret += ", w = " + std::to_string(weight);
    
    if (diffScore.getInclusive() > 0)
      ret += ", DiffScore = " + std::to_string((long)diffScore.getInclusive())
              + "/" + std::to_string((long)diffScore.getExclusive());
    ret += "\n";
    
    return ret;
  }
  
  string TCTATraceNode::toString(int maxDepth, Time minDuration) const {
    string ret = TCTANode::toString(maxDepth, minDuration);
    
    if (depth >= maxDepth) return ret;
    if (minDuration > 0 && time.getDuration() < minDuration) return ret;
    if (name.find("<unknown procedure>") != string::npos) return ret;
    
    for (auto it = children.begin(); it != children.end(); it++)
      ret += (*it)->toString(maxDepth, minDuration);
    return ret;
  }
  
  string TCTLoopNode::toString(int maxDepth, Time minDuration) const {
    if (accept()) {
      string ret = profileNode->toString(maxDepth, minDuration);
      
      if (clusterNode != NULL)
        ret += clusterNode->toString(maxDepth, minDuration);
      
      if (rejectedIterations != NULL)
        ret += rejectedIterations->toString(maxDepth, minDuration);
                
      return ret;
    }
    else
      return profileNode->toString(maxDepth, minDuration);
  }
  
  string TCTLoopClusterNode::toString(int maxDepth, Time minDuration) const {
    string prefix = "";
    for (int i = 0; i <= depth; i++) prefix += "  ";
    
    string ret = prefix + "# clusters = " + std::to_string(numClusters) + "\n";
    for (int i = 0; i < numClusters; i++) {
      ret += prefix + "CLUSTER #" + std::to_string(i) + " members: " + clusters[i].members->toString() + "\n";
      ret += clusters[i].representative->toString(maxDepth, minDuration);
    }
    
    ret += prefix + "Diff ratio among clusters:\n";
    for (int i = 0; i < numClusters; i++) {
      ret += prefix;
      for (int j = 0; j < numClusters; j++)
        ret += std::to_string(diffRatio[i][j] * 100) + "%\t";
      ret += "\n";
    }
    return ret;
  }
  
  string TCTProfileNode::toString(int maxDepth, Time minDuration) const {
    string ret = TCTANode::toString(maxDepth, minDuration);
    
    if (depth >= maxDepth) return ret;
    if (minDuration > 0 && time.getDuration() < minDuration) return ret;
    if (name.find("<unknown procedure>") != string::npos) return ret;
    
    for (auto it = childMap.begin(); it != childMap.end(); it++)
      ret += it->second->toString(maxDepth, minDuration);
    return ret;
  }
  
  TCTLoopNode::TCTLoopNode(int id, string name, int depth, CFGAGraph* cfgGraph) :
    TCTANode(Loop, id, 0, name, depth, cfgGraph, cfgGraph == NULL ? 0 : cfgGraph->vma) {
      numIteration = 0;
      numAcceptedIteration = 0;
      pendingIteration = NULL;
      rejectedIterations = NULL;
      clusterNode = new TCTLoopClusterNode(*this);
      profileNode = NULL;
    }
  
  TCTLoopNode::TCTLoopNode(const TCTLoopNode& orig) : TCTANode(orig) {
    numIteration = orig.numIteration;
    numAcceptedIteration = orig.numAcceptedIteration;
    
    if (orig.pendingIteration != NULL)
      pendingIteration = (TCTIterationTraceNode*)(orig.pendingIteration->duplicate());
    else
      pendingIteration = NULL;
    
    if (orig.rejectedIterations != NULL)
      rejectedIterations = (TCTProfileNode*)(orig.rejectedIterations->duplicate());
    else
      rejectedIterations = NULL;

#ifdef KEEP_ACCEPTED_ITERATION
    for (auto it = orig.acceptedIterations.begin(); it != orig.acceptedIterations.end(); it++)
      acceptedIterations.push_back((TCTIterationTraceNode*)((*it)->duplicate()));
#endif
    
    if (orig.clusterNode != NULL) 
      clusterNode = (TCTLoopClusterNode*)(orig.clusterNode->duplicate());
    else
      clusterNode = NULL;
    
    if (orig.profileNode != NULL)
      profileNode = (TCTProfileNode*)(orig.profileNode->duplicate());
    else
      profileNode = NULL;
  }
  
  TCTLoopNode::TCTLoopNode(const TCTLoopNode& loop1, long weight1, const TCTLoopNode& loop2, long weight2, bool accumulate) : TCTANode(loop1) {
    time.setAsAverageTime(loop1.getTime(), weight1, loop2.getTime(), weight2);
    weight = weight1 + weight2;
    //TODO: plm
    
    numIteration = loop1.numIteration + loop2.numIteration;
    numAcceptedIteration = loop1.numAcceptedIteration + loop2.numAcceptedIteration;
    pendingIteration = NULL;
    
    profileNode = localDQ.mergeProfileNode(loop1.getProfileNode(), weight1, loop2.getProfileNode(), weight2, accumulate, false);
    
    TCTProfileNode* rej1 = loop1.rejectedIterations;
    TCTProfileNode* rej2 = loop2.rejectedIterations;
    if (rej1 == NULL && rej2 == NULL)
      rejectedIterations = NULL;
    else {
      if (rej1 == NULL)  {
        rej1 = (TCTProfileNode*)rej2->voidDuplicate();
        rej1->setWeight(weight1);
      }
      if (rej2 == NULL) {
        rej2 = (TCTProfileNode*)rej1->voidDuplicate();
        rej2->setWeight(weight2);
      }
      rejectedIterations = localDQ.mergeProfileNode(rej1, weight1, rej2, weight2, accumulate, false);
      
      if (loop1.rejectedIterations == NULL) delete rej1;
      if (loop2.rejectedIterations == NULL) delete rej2;
    }
    
    if (loop1.accept() && loop2.accept()) 
      clusterNode = new TCTLoopClusterNode(*loop1.getClusterNode(), *loop2.getClusterNode());
    else
      clusterNode = NULL;
  }
  
  TCTLoopNode::~TCTLoopNode()  {
    if (pendingIteration != NULL) delete pendingIteration;
    if (rejectedIterations != NULL) delete rejectedIterations;
#ifdef KEEP_ACCEPTED_ITERATION
    for (auto it = acceptedIterations.begin(); it != acceptedIterations.end(); it++)
      delete (*it);
#endif
    if (clusterNode != NULL) delete clusterNode;
    if (profileNode != NULL) delete profileNode;
  }
  
  bool TCTLoopNode::acceptPendingIteration() {
    if (pendingIteration == NULL) return false;
    
    int countFunc = 0;
    int countLoop = 0;

    for (int k = 0; k < pendingIteration->getNumChild(); k++)
      if (pendingIteration->getChild(k)->getDuration() / getSamplingPeriod() 
              >= ITER_CHILD_DUR_ACC) {
        if (pendingIteration->getChild(k)->type == TCTANode::Func) countFunc++;
        else countLoop++; // TCTANode::Loop and TCTANode::Prof are all loops.
      }

    // When an iteration has no func child passing the IterationChildDurationThreshold,
    // and has less than two loop children passing the IterationChildDurationThreshold,
    // and the number of children doesn't pass the IterationNumChildThreshold,
    // children of this iteration may belong to distinct iterations in the execution.
    if (countFunc < ITER_NUM_FUNC_ACC 
            && countLoop < ITER_NUM_LOOP_ACC 
            && pendingIteration->getNumChild() < ITER_NUM_CHILD_ACC)
      return false;
    else
      return true;
  }
  
  void TCTLoopNode::finalizePendingIteration() {
    if (pendingIteration == NULL) return;
    
    pendingIteration->finalizeEnclosingLoops();
    pendingIteration->shiftTime(-pendingIteration->getTime().getStartTimeExclusive());
    numIteration++;
    
    if (acceptPendingIteration()) {
      numAcceptedIteration++;
#ifdef KEEP_ACCEPTED_ITERATION      
      acceptedIterations.push_back(pendingIteration->duplicate());
#endif     
      clusterNode->addChild(pendingIteration);
    }
    else {
      TCTProfileNode* prof = TCTProfileNode::newProfileNode(pendingIteration);
      if (rejectedIterations != NULL) {
        rejectedIterations->merge(prof);
        delete prof;
      } else {
        rejectedIterations = prof;
        rejectedIterations->setName("Rejected Iterations");
      }
      delete pendingIteration;
    }
    pendingIteration = NULL;
  }
  
  bool TCTLoopNode::accept() const {
    if (clusterNode == NULL) return false;
    if (rejectedIterations == NULL) return true;
    
    if (rejectedIterations->getDuration() > getDuration() * LOOP_REJ_THRESHOLD)
      return false;
    return true;
  }

  void TCTLoopNode::finalizeEnclosingLoops() {
    finalizePendingIteration();
    
    if (profileNode != NULL)
      print_msg(MSG_PRIO_MAX, "ERROR: TCTLoopNode::finalizeEnclosingLoops() called when profileNode is not NULL for loop %s.\n", getName().c_str());
    profileNode = TCTProfileNode::newProfileNode(this);
    profileNode->clearDiffScore();
    
    if (clusterNode != NULL) {
      clusterNode->getTime().setStartTime(getTime().getStartTimeExclusive(), getTime().getStartTimeInclusive());
      clusterNode->getTime().setEndTime(getTime().getEndTimeInclusive(), getTime().getEndTimeExclusive());
    }
    
    if (accept() && getNumIteration() > 1) {
      print_msg(MSG_PRIO_NORMAL, "\nLoop accepted: \n%s", toString(getDepth()+5, 0).c_str());

      if (getClusterNode()->getNumClusters() > 1) {
        TCTANode* highlight = localDQ.mergeNode(getClusterNode()->getClusterRepAt(0), getClusterNode()->getClusterRepAt(0)->getWeight(), 
                                                getClusterNode()->getClusterRepAt(1), getClusterNode()->getClusterRepAt(1)->getWeight(), false, false);
        print_msg(MSG_PRIO_LOW, "Compare first two clusters:\n");
        print_msg(MSG_PRIO_LOW, "%s\n", highlight->toString(highlight->getDepth()+5, 0).c_str());
        delete highlight;
        print_msg(MSG_PRIO_LOW, "\n");
      }
    }
  }
  
  void TCTLoopNode::shiftTime(Time offset) {
    TCTANode::shiftTime(offset);
    if (clusterNode != NULL) clusterNode->shiftTime(offset);
    if (profileNode != NULL) profileNode->shiftTime(offset);
  }
  
  TCTLoopClusterNode::TCTLoopClusterNode(const TCTLoopClusterNode& other, bool isVoid) : TCTANode(other) {
    if (isVoid) {
      numClusters = 0;
    }
    else {
      numClusters = other.numClusters;
      for (int i = 0; i < numClusters; i++) {
        clusters[i].representative = other.clusters[i].representative->duplicate();
        clusters[i].members = other.clusters[i].members->duplicate();
      }
      for (int i = 0; i < numClusters; i++)
        for (int j = i+1; j < numClusters; j++)
          diffRatio[i][j] = other.diffRatio[i][j];
    }
  }

  TCTLoopClusterNode::TCTLoopClusterNode(const TCTLoopClusterNode& cluster1, const TCTLoopClusterNode& cluster2) : TCTANode(cluster1) {
    getTime().setAsAverageTime(cluster1.getTime(), cluster1.getWeight(), cluster2.getTime(), cluster1.getWeight());
    weight = cluster1.weight + cluster2.weight;
    
    numClusters = cluster1.numClusters + cluster2.numClusters;
    
    for (int i = 0; i < cluster1.numClusters; i++) {
      clusters[i].representative = cluster1.clusters[i].representative->duplicate();
      clusters[i].members = cluster1.clusters[i].members->duplicate();
    }
    for (int i = 0; i < cluster2.numClusters; i++) {
      clusters[i+cluster1.numClusters].representative = cluster2.clusters[i].representative->duplicate();
      clusters[i+cluster1.numClusters].members = cluster2.clusters[i].members->duplicate();
    }
    
    for (int i = 0; i < cluster1.numClusters; i++)
      for (int j = i+1; j < cluster1.numClusters; j++)
        diffRatio[i][j] = cluster1.diffRatio[i][j];
    for (int i = 0; i < cluster2.numClusters; i++)
      for (int j = i+1; j < cluster2.numClusters; j++)
        diffRatio[i+cluster1.numClusters][j+cluster1.numClusters] = cluster2.diffRatio[i][j];
    for (int i = 0; i < cluster1.numClusters; i++)
      for (int j = 0; j < cluster2.numClusters; j++) {
        TCTANode* tempNode = localDQ.mergeNode(clusters[i].representative, 1, clusters[j+cluster1.numClusters].representative, 1, false, true);
        diffRatio[i][j+cluster1.numClusters] = tempNode->getDiffScore().getInclusive() / 
                (double)(clusters[i].representative->getDuration() + clusters[j+cluster1.numClusters].representative->getDuration());
        delete tempNode;
      }
    
    while (numClusters > 1 && mergeClusters());
  }
  
  void TCTLoopClusterNode::addChild(TCTANode* child) {
    clusters[numClusters].representative = child;
    child->setName("CLUSTER_#" + std::to_string(numClusters) + " REP");
    clusters[numClusters].members = new TCTClusterMembers();
    clusters[numClusters].members->addMember(((TCTIterationTraceNode*)child)->iterNum);
    
    for (int k = 0; k < numClusters; k++) {
      TCTANode* tempNode = localDQ.mergeNode(child, 1, clusters[k].representative, 1, false, true);
      diffRatio[k][numClusters] = tempNode->getDiffScore().getInclusive() / (double)(child->getDuration() + clusters[k].representative->getDuration());
      delete tempNode;
    }
    
    numClusters++;
    while (numClusters > 1 && mergeClusters());
  }
  
  bool TCTLoopClusterNode::mergeClusters() {
    int min_x = 0;
    int min_y = 1;
    double maxDR = diffRatio[min_x][min_y];
    
    for (int x = 0; x < numClusters; x++)
      for (int y = x+1; y < numClusters; y++)
        if (diffRatio[x][y] < diffRatio[min_x][min_y]) {
          min_x = x;
          min_y = y;
        }
        else if (diffRatio[x][y] > maxDR)
          maxDR = diffRatio[x][y];
    
    if (diffRatio[min_x][min_y] <= MIN_DIFF_RATIO ||
          diffRatio[min_x][min_y] <= maxDR * REL_DIFF_RATIO ||
          numClusters > MAX_NUM_CLUSTER) {
      TCTANode* tempNode = localDQ.mergeNode(clusters[min_x].representative, clusters[min_x].representative->getWeight(),
              clusters[min_y].representative, clusters[min_y].representative->getWeight(), true, false);
      delete clusters[min_x].representative;
      delete clusters[min_y].representative;
      clusters[min_x].representative = tempNode;
      tempNode->setName("CLUSTER_#" + std::to_string(min_x) + " REP");
      
      TCTClusterMembers* tempMembers = new TCTClusterMembers(*clusters[min_x].members, *clusters[min_y].members);
      delete clusters[min_x].members;
      delete clusters[min_y].members;
      clusters[min_x].members = tempMembers;
      
      // Remove the cluster at min_y. Update clusters[], diffRatio[][], and numClusters.
      for (int y = min_y+1; y < numClusters; y++) {
        clusters[y-1] = clusters[y];
        for (int x = 0; x < min_y; x++)
          diffRatio[x][y-1] = diffRatio[x][y];
        for (int x = min_y + 1; x < y; x++)
          diffRatio[x-1][y-1] = diffRatio[x][y];
      }
      numClusters--;
      clusters[numClusters].members = NULL;
      clusters[numClusters].representative = NULL;
      
      // Recompute diffRatio values for the merged cluster.
      for (int k = 0; k < min_x; k++) {
        TCTANode* tempNode = localDQ.mergeNode(clusters[min_x].representative, 1, clusters[k].representative, 1, false, true);
        diffRatio[k][min_x] = tempNode->getDiffScore().getInclusive() / (double)(clusters[min_x].representative->getDuration() + clusters[k].representative->getDuration());
        delete tempNode;
      }
      
      for (int k = min_x+1; k < numClusters; k++) {
        TCTANode* tempNode = localDQ.mergeNode(clusters[min_x].representative, 1, clusters[k].representative, 1, false, true);
        diffRatio[min_x][k] = tempNode->getDiffScore().getInclusive() / (double)(clusters[min_x].representative->getDuration() + clusters[k].representative->getDuration());
        delete tempNode;
      }
      
      return true;
    }
    
    return false;
  }
  
  TCTProfileNode::TCTProfileNode(const TCTATraceNode& trace) : TCTANode (trace, Prof) {
    for (int k = 0; k < trace.getNumChild(); k++) {
      TCTProfileNode* child = TCTProfileNode::newProfileNode(trace.getChild(k));
      child->getTime().toProfileTime();
      if (childMap.find(child->id) == childMap.end())
        childMap[child->id] = child;
      else {
        childMap[child->id]->merge(child);
        delete child;
      }
    }
  }
  
  TCTProfileNode::TCTProfileNode(const TCTLoopNode& loop) : TCTANode (loop, Prof) {
    /*
    for (auto it = loop.acceptedIterations.begin(); it != loop.acceptedIterations.end(); it++) {
      TCTProfileNode* child = TCTProfileNode::newProfileNode(*it);
      child->setDepth(this->depth);
      for (auto iit = child->childMap.begin(); iit != child->childMap.end(); iit++) {
        if (childMap.find(iit->second->id) == childMap.end())
          childMap[iit->second->id] = (TCTProfileNode*) iit->second->duplicate();
        else
          childMap[iit->second->id]->merge(iit->second);
      }
      delete child;
    }*/
    if (loop.getClusterNode() != NULL) {
      for (int k = 0; k < loop.getClusterNode()->getNumClusters(); k++) {
        TCTProfileNode* child = TCTProfileNode::newProfileNode(loop.getClusterNode()->getClusterRepAt(k));
        child->setDepth(this->depth);
        child->amplify(child->weight/weight);
        for (auto iit = child->childMap.begin(); iit != child->childMap.end(); iit++) {
          if (childMap.find(iit->second->id) == childMap.end())
            childMap[iit->second->id] = (TCTProfileNode*) iit->second->duplicate();
          else
            childMap[iit->second->id]->merge(iit->second);
        }
        delete child;
      }
    }
    
    if (loop.getRejectedIterations() != NULL) {
      TCTProfileNode* child = TCTProfileNode::newProfileNode(loop.getRejectedIterations());
      child->setDepth(this->depth);
      for (auto iit = child->childMap.begin(); iit != child->childMap.end(); iit++) {
        if (childMap.find(iit->second->id) == childMap.end())
          childMap[iit->second->id] = (TCTProfileNode*) iit->second->duplicate();
        else
          childMap[iit->second->id]->merge(iit->second);
      }
      delete child;
    }
    
    //if (loop.pendingIteration != NULL)
      //print_msg(MSG_PRIO_HIGH, "ERROR: Loop %s converted to profile with pending iteration not been processed.\n", loop.getName().c_str());
  }
  
  TCTProfileNode::TCTProfileNode(const TCTProfileNode& prof, bool copyChildMap) : TCTANode (prof) {
    if (copyChildMap)
      for (auto it = prof.childMap.begin(); it != prof.childMap.end(); it++)
        childMap[it->second->id] = (TCTProfileNode*) it->second->duplicate();
  }
  
  void TCTProfileNode::addChild(TCTANode* child) {
    TCTProfileNode* profChild = NULL; 
    if (child->type == TCTANode::Prof)
      profChild = (TCTProfileNode*) child;
    else {
      profChild = TCTProfileNode::newProfileNode(child);
      delete child;
    }

    if (childMap.find(profChild->id) == childMap.end())
      childMap[profChild->id] = profChild;
    else {
      childMap[profChild->id]->merge(profChild);
      delete profChild;
    }
  }
  
  void TCTProfileNode::merge(const TCTProfileNode* other) {
    time.addTime(other->time);
    for (auto it = other->childMap.begin(); it != other->childMap.end(); it++) {
      if (childMap.find(it->second->id) == childMap.end())
        childMap[it->second->id] = (TCTProfileNode*) it->second->duplicate();
      else
        childMap[it->second->id]->merge(it->second);
    }
  }
  
  void TCTProfileNode::getExclusiveDuration(Time& minExclusive, Time& maxExclusive) const {
    minExclusive = getMinDuration();
    maxExclusive = getMaxDuration();
    for (auto it = childMap.begin(); it != childMap.end(); it++) {
      TCTProfileNode* child = it->second;
      minExclusive -= child->getMaxDuration();
      maxExclusive -= child->getMinDuration();
    }
  }
  
  void TCTProfileNode::amplify(long amplifier) {
    Time minDuration = this->getMinDuration();
    Time maxDuration = this->getMaxDuration();
    getTime().setDuration(minDuration * amplifier, maxDuration * amplifier);
    
    for (auto it = childMap.begin(); it != childMap.end(); it++)
      it->second->amplify(amplifier);
    
    setWeight(weight/amplifier);
  }
  
  void TCTATraceNode::getLastChildEndTime(int idx, Time& inclusive, Time& exclusive) const {
    if (idx < 0 || idx > getNumChild()) {
      print_msg(MSG_PRIO_MAX, "ERROR: wrong idx %d for getLastChildEndTime(). 0 <= idx <= %d should hold.\n", 
              idx, getNumChild());
      return;
    }

    if (idx == 0) {
      inclusive = time.getStartTimeExclusive();
      exclusive = time.getStartTimeInclusive();
    }
    else {
      inclusive = getChild(idx-1)->getTime().getEndTimeInclusive();
      exclusive = getChild(idx-1)->getTime().getEndTimeExclusive();
    }
  }
  
  void TCTATraceNode::getCurrChildStartTime(int idx, Time& exclusive, Time& inclusive) const {
    if (idx < 0 || idx > getNumChild()) {
      print_msg(MSG_PRIO_MAX, "ERROR: wrong idx %d for getCurrChildStartTime(). 0 <= idx <= %d should hold.\n", 
              idx, getNumChild());
      return;
    }
    
    if (idx == getNumChild()) {
      exclusive = time.getEndTimeInclusive();
      inclusive = time.getEndTimeExclusive();
    }
    else {
      exclusive = getChild(idx)->getTime().getStartTimeExclusive();
      inclusive = getChild(idx)->getTime().getStartTimeInclusive();
    }
  }
  
  void TCTATraceNode::getGapBeforeChild(int idx, Time& minGap, Time& maxGap) const {
    if (idx < 0 || idx > getNumChild()) {
      print_msg(MSG_PRIO_MAX, "ERROR: wrong idx %d for getGapBeforeChild(). 0 <= idx <= %d should hold.\n", 
              idx, getNumChild());
      return;
    }

    Time lastEndInclusive, lastEndExclusive, currStartExclusive, currStartInclusive;
    getLastChildEndTime(idx, lastEndInclusive, lastEndExclusive);
    getCurrChildStartTime(idx, currStartExclusive, currStartInclusive);
    
    minGap = currStartExclusive - lastEndExclusive + 1;
    maxGap = currStartInclusive - lastEndInclusive - 1;
  }
}